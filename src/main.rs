use std::time::Instant;

//use flink::{f32x4, f32x4x4, vec3, vec4};
use glutin::event::{DeviceEvent, Event, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};

mod input;
mod image;
mod camera;
mod background;
mod particles;

fn main() -> anyhow::Result<()> {
    unsafe {
        let el = EventLoop::new();
        let wb = glutin::window::WindowBuilder::new()
            .with_title("rust - gpu - snippets")
            .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
        let window = glutin::ContextBuilder::new()
            .with_srgb(true)
            .with_multisampling(4)
            .build_windowed(wb, &el)?
            .make_current()
            .unwrap();

        let grr = grr::Device::new(
            |symbol| window.get_proc_address(symbol) as *const _,
            grr::Debug::Disable,
        );

        let begin = Instant::now();
        let mut camera = camera::Camera::new(0.40, 5.0);
        let mut input = input::Input::new();

        // Modules
        let mut background = background::Background::new(&grr)?;
        let mut particles = particles::Particles::new(&grr, particles::Mode::Field)?;
        
        el.run(move |event, _, control_flow| {

            match event {
                Event::LoopDestroyed => return,
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(physical_size) => {
                        window.resize(physical_size);
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                Event::DeviceEvent { event, .. } => match event {
                    DeviceEvent::MouseMotion { delta } => {
                        input.update_mouse_motion((delta.0 as _, delta.1 as _));
                    }
                    DeviceEvent::Button { state, .. } => {
                        input.update_mouse(state);
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    let time = begin.elapsed().as_secs_f32();
                    let size = window.window().inner_size();
                    camera.update(&grr, &input, size.width as f32, size.height as f32, time);
                    input.reset_delta();
        
                    grr.bind_framebuffer(grr::Framebuffer::DEFAULT);
                    grr.clear_attachment(
                        grr::Framebuffer::DEFAULT,
                        grr::ClearAttachment::ColorFloat(0, [0.0, 0.0, 0.0, 1.0]),
                    );
                    grr.clear_attachment(
                        grr::Framebuffer::DEFAULT,
                        grr::ClearAttachment::Depth(1.0),
                    );
        
                    // modules
                    background.update(&grr, &camera, &input, time);
                    particles.update(&grr, &camera, &input, time);

                    window.swap_buffers().unwrap();
                },
                _ => ()
            }
        })
    }
}
