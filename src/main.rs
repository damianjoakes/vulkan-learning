mod engine;
mod init;

use crate::engine::VulkanEngine;
use winit::event_loop::{ControlFlow, EventLoop};

pub const VALIDATION_ENABLED: bool = true;

fn main() {
    if VALIDATION_ENABLED == true {
        unsafe { std::env::set_var("RUST_LOG", "debug") };
    }

    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut engine = VulkanEngine::new();

    event_loop.run_app(&mut engine).unwrap()
}
