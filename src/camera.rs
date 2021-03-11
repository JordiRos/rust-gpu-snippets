use flink::{f32x4x4, vec3, Vec3};

use crate::input;

pub struct Camera {
    fov: f32,
    distance: f32,
    position: Vec3<f32>,
    world_view: f32x4x4,
    view_proj: f32x4x4,
    world_view_inv: f32x4x4,
    view_proj_inv: f32x4x4,    
}

impl Camera {
    pub fn new(fov: f32, distance: f32) -> Self {
        Camera {
            fov: fov,
            distance: distance,
            position: vec3(0.0, 0.0, distance),
            world_view: f32x4x4::look_at_inv(vec3(0.0, 0.0, distance), vec3(0.0, 0.0, 1.0)),
            view_proj: f32x4x4::perspective(
                std::f32::consts::PI * 0.25,
                1.0,
                0.1,
                10000.0,
            ),
            world_view_inv: f32x4x4::look_at_inv(vec3(0.0, 0.0, distance), vec3(0.0, 0.0, 1.0)),
            view_proj_inv: f32x4x4::perspective(
                std::f32::consts::PI * 0.25,
                1.0,
                0.1,
                10000.0,
            ),            
        }
    }

    pub fn update(&mut self, grr: &grr::Device, input: &input::Input, width: f32, height: f32, _time: f32) {
        let aspect = width / height;
        self.position.x = (input.mouse_pos().x / width + 0.5) * 10.0;
        self.position.y = (input.mouse_pos().y / height + 0.5) * 2.0;
        self.position.z = self.distance;

        self.world_view = f32x4x4::look_at(self.position, self.position);
        self.view_proj = f32x4x4::perspective(
            std::f32::consts::PI * self.fov,
            aspect,
            0.1,
            10000.0,
        );
        self.world_view_inv = f32x4x4::look_at_inv(self.position, self.position);
        self.view_proj_inv = f32x4x4::perspective_inv(
            std::f32::consts::PI * self.fov,
            aspect,
            0.1,
            10000.0,
        );        

        unsafe {
            grr.set_viewport(
                0,
                &[grr::Viewport {
                    x: 0.0,
                    y: 0.0,
                    w: width as _,
                    h: height as _,
                    n: 0.0,
                    f: 1.0,
                }],
            );
            grr.set_scissor(
                0,
                &[grr::Region {
                    x: 0,
                    y: 0,
                    w: width as _,
                    h: height as _,
                }],
            );
        }
    }

    pub fn world_view(&self) -> f32x4x4 {
        self.world_view
    }

    pub fn view_proj(&self) -> f32x4x4 {
        self.view_proj
    }

    pub fn world_view_inv(&self) -> f32x4x4 {
        self.world_view_inv
    }

    pub fn view_proj_inv(&self) -> f32x4x4 {
        self.view_proj_inv
    }
}
