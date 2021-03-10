use anyhow::{Result};
use std::mem;
use flink::{f32x4x4};

use crate::input;
use crate::image;
use crate::camera;

pub enum Mode {
    Lines,
}

const NUM_PARTICLES: usize = 20000;
const BUFFER_STRIDE: u64 = (mem::size_of::<f32>() * 4) as u64;

#[repr(C)]
struct LocalsParticles {
    world_view: f32x4x4,
    view_proj: f32x4x4,
    time: f32,
}

pub struct Particles {
    mode: Mode,
    texture: grr::Image,
    vertices: grr::Buffer,
    positions: grr::Buffer,
    pipeline: grr::Pipeline,
    vertex_array: grr::VertexArray,
    sampler: grr::Sampler,
    num_particles: u32,
}

impl Particles {
    pub fn new(grr: &grr::Device, mode: Mode) -> Result<Self> {
        unsafe {
            let spirv = include_bytes!(env!("shader.spv"));
            let texture = image::load_png("assets/particle.png", grr, grr::Format::R8G8B8A8_SRGB, true).unwrap();

            let vs = grr.create_shader(
                grr::ShaderStage::Vertex,
                grr::ShaderSource::Spirv {
                    entrypoint: "particles_vs",
                },
                &spirv[..],
                grr::ShaderFlags::VERBOSE,
            ).unwrap();

            let fs = grr.create_shader(
                grr::ShaderStage::Fragment,
                grr::ShaderSource::Spirv {
                    entrypoint: "particles_fs",
                },
                &spirv[..],
                grr::ShaderFlags::VERBOSE,
            ).unwrap();

            let pipeline = grr.create_graphics_pipeline(
                grr::VertexPipelineDesc {
                    vertex_shader: vs,
                    tessellation_control_shader: None,
                    tessellation_evaluation_shader: None,
                    geometry_shader: None,
                    fragment_shader: Some(fs),
                },
                grr::PipelineFlags::VERBOSE,
            ).unwrap();

            let vertex_array = grr.create_vertex_array(&[
                grr::VertexAttributeDesc {
                    location: 0,
                    binding: 0,
                    format: grr::VertexFormat::Xyz32Float,
                    offset: 0,
                },
                grr::VertexAttributeDesc {
                    location: 1,
                    binding: 1,
                    format: grr::VertexFormat::Xyzw32Float,
                    offset: 0,
                },
            ]).unwrap();

            let sampler = grr.create_sampler(grr::SamplerDesc {
                min_filter: grr::Filter::Linear,
                mag_filter: grr::Filter::Linear,
                mip_map: None,
                address: (
                    grr::SamplerAddress::ClampBorder,
                    grr::SamplerAddress::ClampBorder,
                    grr::SamplerAddress::ClampBorder,
                ),
                lod_bias: 0.0,
                lod: 0.0..10.0,
                compare: None,
                border_color: [0.0, 0.0, 0.0, 0.0],
            }).unwrap();

            let vertices: [f32; 12] = [-0.5,-0.5, 0.0, 0.5,-0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.0];
            let vertices = grr.create_buffer_from_host(
                grr::as_u8_slice(&vertices),
                grr::MemoryFlags::DEVICE_LOCAL,
            )
            .unwrap();

            let positions = grr.create_buffer(
                BUFFER_STRIDE * NUM_PARTICLES as u64,
                grr::MemoryFlags::DEVICE_LOCAL | grr::MemoryFlags::CPU_MAP_WRITE,
            )
            .unwrap();            

            Ok(Particles {
                mode: mode,
                texture: texture,
                vertices: vertices,
                positions: positions,
                pipeline: pipeline,
                vertex_array: vertex_array,
                sampler: sampler,
                num_particles: NUM_PARTICLES as u32,
            })
        }
    }

    pub fn update(&mut self, grr: &grr::Device, camera: &camera::Camera, _input: &input::Input, time: f32) {
        unsafe {
            let num_particles = self.num_particles;
            let buffer_size = BUFFER_STRIDE * num_particles as u64;

            match self.mode {
                Mode::Lines => {
                    let positions = grr.map_buffer::<f32>(self.positions, 0..buffer_size, grr::MappingFlags::UNSYNCHRONIZED);
                    for i in 0..num_particles {
                        let t = time * 2.0;
                        let group = 0;
                        let idx = (i * 4) as usize;
                        let f = i as f32;
                        let ang = t * 0.25 + (i as f32) * 0.005;
                        let offx = (group as f32 + f * 0.01 + t * 0.25).sin() * 0.25;
                        let offz = (group as f32 + f * 0.006 + t * 0.2).sin() * 0.22;
        
                        positions[idx + 0] = ang.sin() * 1.0 + offx;
                        positions[idx + 1] = -1.5 + f * 0.00015;
                        positions[idx + 2] = ang.cos() * 1.0 + offz;
                        positions[idx + 3] = 0.03 + (f as f32 * 0.01).sin() * 0.01;
                    }
                    grr.unmap_buffer(self.positions);
                },
            }

            // render
            let color_blend = grr::ColorBlend {
                attachments: vec![grr::ColorBlendAttachment {
                    blend_enable: true,
                    color: grr::BlendChannel {
                        src_factor: grr::BlendFactor::SrcAlpha,
                        dst_factor: grr::BlendFactor::One,
                        blend_op: grr::BlendOp::Add,
                    },
                    alpha: grr::BlendChannel {
                        src_factor: grr::BlendFactor::SrcAlpha,
                        dst_factor: grr::BlendFactor::One,
                        blend_op: grr::BlendOp::Add,
                    },
                }],
            };
            
            let state_ds = grr::DepthStencil {
                depth_test: true,
                depth_write: false,
                depth_compare_op: grr::Compare::LessEqual,
                stencil_test: false,
                stencil_front: grr::StencilFace::KEEP,
                stencil_back: grr::StencilFace::KEEP,            
            };            

            // particles
            let locals = LocalsParticles {
                world_view: camera.world_view_inv(),
                view_proj: camera.view_proj(),
                time: time,
            };

            let u_locals = grr.create_buffer_from_host(
                grr::as_u8_slice(&[locals]),
                grr::MemoryFlags::DEVICE_LOCAL,
            )
            .unwrap();

            grr.bind_pipeline(self.pipeline);
            grr.bind_depth_stencil_state(&state_ds);
            grr.bind_color_blend_state(&color_blend);
            grr.bind_vertex_array(self.vertex_array);
            grr.bind_vertex_buffers(
                self.vertex_array,
                0,
                &[
                    grr::VertexBufferView {
                        buffer: self.vertices,
                        offset: 0,
                        stride: (3 * mem::size_of::<f32>()) as _,
                        input_rate: grr::InputRate::Vertex,
                    },
                    grr::VertexBufferView {
                        buffer: self.positions,
                        offset: 0,
                        stride: (4 * mem::size_of::<f32>()) as _,
                        input_rate: grr::InputRate::Instance { divisor: 1 },
                    }
                ],
            );            
            grr.bind_uniform_buffers(
                0,
                &[grr::BufferRange {
                    buffer: u_locals,
                    offset: 0,
                    size: std::mem::size_of::<LocalsParticles>() as _,
                }],
            );
            grr.bind_image_views(
                0,
                &[
                    self.texture.as_view(),
                ],
            );
            grr.bind_samplers(0, &[self.sampler]);
            grr.draw(grr::Primitive::TriangleStrip, 0..4, 0..num_particles);

            // end
            grr.delete_buffer(u_locals);
        }
    }
}
