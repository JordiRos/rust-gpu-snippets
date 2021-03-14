use anyhow::{Result};
use std::mem;
use flink::{f32x4x4, Vec4};

use crate::input;
use crate::image;
use crate::camera;

fn lerp(a: f32, b: f32, v: f32) -> f32 {
    return a + (b - a) * v;
}

fn get_color(colors: &Vec<f32>, value: f32) -> Vec4<f32> {
    let mut color = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    let num = colors.len() / 5;
    for i in 1..num {
        let a = i * 5;
        let b = (i - 1) * 5;
        if value >= colors[b + 4] && value <= colors[a + 4] {
            if i > 0 {
                let f = (value - colors[b + 4]) / (colors[a + 4] - colors[b + 4]);
                color.x = lerp(colors[b], colors[a], f);
                color.y = lerp(colors[b + 1], colors[a + 1], f);
                color.z = lerp(colors[b + 2], colors[a + 2], f);
                color.w = lerp(colors[b + 3], colors[a + 3], f);
                
            }
            else {
                let a = i * 5;
                color.x = colors[a];
                color.y = colors[a + 1];
                color.z = colors[a + 2];
                color.w = colors[a + 3];
            }
            break;
        }
    }
    return color;
}

const NUM_PARTICLES: usize = 25000;
const BUFFER_STRIDE: u64 = (mem::size_of::<f32>() * 4) as u64;

#[repr(C)]
struct LocalsParticles {
    world_view: f32x4x4,
    view_proj: f32x4x4,
    depth: f32,
    apperture: f32,
}

pub struct Effect {
    texture: grr::Image,
    vertices: grr::Buffer,
    positions: grr::Buffer,
    colors: grr::Buffer,
    pipeline: grr::Pipeline,
    vertex_array: grr::VertexArray,
    sampler: grr::Sampler,
    num_particles: u32,
    first_time: bool,
}

impl Effect {
    pub fn new(grr: &grr::Device) -> Result<Self> {
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
                grr::VertexAttributeDesc {
                    location: 2,
                    binding: 2,
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

            let colors = grr.create_buffer(
                BUFFER_STRIDE * NUM_PARTICLES as u64,
                grr::MemoryFlags::DEVICE_LOCAL | grr::MemoryFlags::CPU_MAP_WRITE,
            )
            .unwrap();            

            Ok(Effect {
                texture: texture,
                vertices: vertices,
                positions: positions,
                colors: colors,
                pipeline: pipeline,
                vertex_array: vertex_array,
                sampler: sampler,
                num_particles: NUM_PARTICLES as u32,
                first_time: true,
            })
        }
    }

    pub fn update(&mut self, grr: &grr::Device, camera: &camera::Camera, _input: &input::Input, time: f32) {
        unsafe {
            let num_particles = self.num_particles;
            let buffer_size = BUFFER_STRIDE * num_particles as u64;

            // line mode
            let b = 7.0;
            let purple_colour_scheme: Vec<f32> = vec![
                0.17 * b, 0.0 * b, 0.83 * b, 1.0 * b, 0.0,
                1.0 * b, 0.27 * b, 0.41 * b, 1.0 * b, 1.0,
            ];
            let positions = grr.map_buffer::<f32>(self.positions, 0..buffer_size, grr::MappingFlags::UNSYNCHRONIZED);
            let colors = grr.map_buffer::<f32>(self.colors, 0..buffer_size, grr::MappingFlags::UNSYNCHRONIZED);
            for i in 0..num_particles {
                let t = time * 2.0;
                let idx = (i * 4) as usize;
                let f = i as f32;
                let ang = t * 0.25 + (i as f32) * 0.005;
                let offx = (f * 0.012 + t * 0.85).sin() * 0.31;
                let offz = (f * 0.015 + t * 1.32).sin() * 0.26;

                positions[idx + 0] = ang.sin() * 2.0 + offx;
                positions[idx + 1] = -3.5 + f * 0.0003;
                positions[idx + 2] = ang.cos() * 2.0 + offz;
                positions[idx + 3] = 0.05;
                
                if self.first_time {
                    let v  = (idx as f32 * 2.3).sin() * 0.5 + 0.5;
                    let color = get_color(&purple_colour_scheme, v);
                    colors[idx + 0] = color.x;
                    colors[idx + 1] = color.y;
                    colors[idx + 2] = color.z;
                    colors[idx + 3] = color.w;
                }
            }
            grr.unmap_buffer(self.positions);
            grr.unmap_buffer(self.colors);
            self.first_time = false;

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
                depth: -3.5,
                apperture: 0.02,
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
                    },
                    grr::VertexBufferView {
                        buffer: self.colors,
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
            grr.draw(grr::Primitive::TriangleStrip, 0..4, 0..(num_particles * 4) as u32);

            // end
            grr.delete_buffer(u_locals);
        }
    }
}

