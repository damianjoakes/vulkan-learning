mod engine;
mod init;
mod shaders;

use std::ffi::{c_char, CStr};
use vulkanalia::vk;
use vulkanalia::vk::StringArray;
use crate::engine::VulkanEngine;
use winit::event_loop::{ControlFlow, EventLoop};

pub const VALIDATION_ENABLED: bool = true;
pub const REQUIRED_EXTENSIONS: [StringArray<256>; 1] = [
    vk::KHR_SWAPCHAIN_EXTENSION.name
];


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
