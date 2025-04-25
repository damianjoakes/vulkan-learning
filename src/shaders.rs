use anyhow::anyhow;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::{DeviceV1_0, HasBuilder};
use vulkanalia::{vk, Device};

/// Wraps shader code loaded via `load_shader` in a ShaderModule, so that it may be loaded into the
/// render pipeline.
pub fn create_shader_module(
    device: &Device,
    code: &[u8],
) -> Result<vk::ShaderModule, anyhow::Error> {
    let bytecode = Bytecode::new(code).map_err(|e| anyhow!(e))?;

    let create_info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| anyhow!(e))
    }
}
