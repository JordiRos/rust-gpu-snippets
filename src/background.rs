use anyhow::{Result};
use flink::{f32x4x4};

use crate::input;
use crate::camera;

#[repr(C)]
struct LocalsBackground {
    world_view: f32x4x4,
    view_proj: f32x4x4,
}

pub struct Background {
    pipeline: grr::Pipeline,
    vertex_array: grr::VertexArray,
    sampler: grr::Sampler,
}

impl Background {
    pub fn new(grr: &grr::Device) -> Result<Self> {
        unsafe {
            let spirv = include_bytes!(env!("shader.spv"));

            let vertex_array = grr.create_vertex_array(&[])?;

            let vs = grr.create_shader(
                grr::ShaderStage::Vertex,
                grr::ShaderSource::Spirv {
                    entrypoint: "background_vs",
                },
                &spirv[..],
                grr::ShaderFlags::VERBOSE,
            ).unwrap();
            
            let fs = grr.create_shader(
                grr::ShaderStage::Fragment,
                grr::ShaderSource::Spirv {
                    entrypoint: "background_fs",
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
            )?;

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

            Ok(Background {
                pipeline: pipeline,
                vertex_array: vertex_array,
                sampler: sampler,
            })
        }
    }

    pub fn update(&mut self, grr: &grr::Device, camera: &camera::Camera, _input: &input::Input, _time: f32) {
        unsafe {
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
                depth_test: false,
                depth_write: false,
                depth_compare_op: grr::Compare::LessEqual,
                stencil_test: false,
                stencil_front: grr::StencilFace::KEEP,
                stencil_back: grr::StencilFace::KEEP,
            };

            let locals = LocalsBackground {
                world_view: camera.world_view(),
                view_proj: camera.view_proj_inv(),
            };
            let u_locals = grr
                .create_buffer_from_host(
                    grr::as_u8_slice(&[locals]),
                    grr::MemoryFlags::DEVICE_LOCAL,
                )
                .unwrap();

            grr.bind_pipeline(self.pipeline);
            grr.bind_depth_stencil_state(&state_ds);
            grr.bind_color_blend_state(&color_blend);
            grr.bind_vertex_array(self.vertex_array);
            grr.bind_uniform_buffers(
                0,
                &[grr::BufferRange {
                    buffer: u_locals,
                    offset: 0,
                    size: std::mem::size_of::<LocalsBackground>() as _,
                }],
            );
            grr.bind_samplers(0, &[self.sampler]);
            grr.draw(grr::Primitive::Triangles, 0..3, 0..1);

            grr.delete_buffer(u_locals);
        }
    }
}
