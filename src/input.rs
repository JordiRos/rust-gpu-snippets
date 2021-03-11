use glutin::event::{ElementState};
use flink::{Vec2};

pub struct Input {
    mouse: ElementState,
    mouse_delta: Vec2<f32>,
    mouse_pos: Vec2<f32>,
}

impl Input {
    pub fn new() -> Self {
        Input {
            mouse: ElementState::Released,
            mouse_delta: Vec2::<f32> { x: 0.0, y: 0.0 },
            mouse_pos: Vec2::<f32> { x: 0.0, y: 0.0 },
        }
    }

    pub fn update_mouse(&mut self, state: ElementState) {
        self.mouse = state;
    }

    pub fn update_mouse_motion(&mut self, (dx, dy): (f64, f64)) {
        if self.mouse == ElementState::Pressed {
            self.mouse_delta.x += dx as f32;
            self.mouse_delta.y += dy as f32;
            self.mouse_pos.x += self.mouse_delta.x;
            self.mouse_pos.y += self.mouse_delta.y;
        }
    }

    /*
    pub fn mouse_delta(&self) -> Vec2::<f32> {
        self.mouse_delta
    }
    */

    pub fn mouse_pos(&self) -> Vec2::<f32> {
        self.mouse_pos
    }
 
    pub fn reset_delta(&mut self) {
        self.mouse_delta = Vec2::<f32> { x: 0.0, y: 0.0 };
    }
}