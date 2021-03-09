#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items)]
#![feature(register_attr)]
#![register_attr(spirv)]

use spirv_std::storage_class::{Input, Output, Uniform, UniformConstant};
use spirv_std::{SampledImage, Image2d, ImageCube};
use flink::{f32x3x3, f32x4x4, f32x2, f32x3, f32x4, vec3, vec4};
use spirv_std::num_traits::Float;

#[spirv(block)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocalsParticles {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn particles_vs(
    #[spirv(vertex_id)] c_vertex_id: Input<i32>,
    #[spirv(instance_id)] c_instance_id: Input<i32>,
    #[spirv(binding = 0)] u_locals: Uniform<LocalsParticles>,
    #[spirv(location = 0)] v_position: Input<f32x3>,
    #[spirv(location = 1)] v_posscale: Input<f32x4>,
    #[spirv(position)] mut a_position: Output<f32x4>,
    #[spirv(location = 0)] mut a_texcoord: Output<f32x3>,
) {
    let instance_id = c_instance_id.load() as f32;
    let locals = u_locals.load();

    let position = v_position.load();
    let posscale = v_posscale.load();
    let scale = posscale.w;

    let mut pos_view = vec4(posscale.x, posscale.y, posscale.z, 1.0) * locals.world_to_view;
    pos_view.x += position.x * scale;
    pos_view.y += position.y * scale;
    pos_view.z += position.z * scale;

    let pos_clip = pos_view * locals.view_to_clip;

    a_position.store(pos_clip);
    a_texcoord.store(vec3(position.x + 0.5, 0.5 - position.y, 0.0));
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn particles_fs(
    #[spirv(binding = 0)] u_texture: UniformConstant<SampledImage<Image2d>>,
    #[spirv(location = 0)] f_texcoord: Input<f32x3>,
    mut output: Output<f32x4>,
) {
    let texture = u_texture.load();
    let texcoord = f_texcoord.load();
    let tex = texture.sample(spirv_std::glam::Vec2::new(texcoord.x, texcoord.y));
    output.store(vec4(tex.x * tex.w, tex.y * tex.w, tex.z * tex.w, tex.w));
}
