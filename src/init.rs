use crate::VALIDATION_ENABLED;
use anyhow::anyhow;
use std::ffi::{c_char, c_void, CStr};
use log::{debug, error, info, warn};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{EntryV1_0, ExtDebugUtilsExtension, HasBuilder};
use vulkanalia::window::get_required_instance_extensions;
use vulkanalia::{vk, Entry, Instance, Version, VkResult};
use winit::window::Window;
use crate::engine::VEngineData;

/// A list of validation layers to use.
pub const VALIDATION_LAYERS: [*const c_char; 1] =
    [b"VK_LAYER_KHRONOS_validation".as_ptr() as *const c_char];

/// Creates a new Vulkan entry using the LibLoadingLoader library. This will automatically
/// resolve the path to the Vulkan SDK install, and include it in the application.
pub unsafe fn create_vulkan_entry() -> Entry {
    Entry::new(LibloadingLoader::new(LIBRARY).unwrap()).unwrap()
}

/// Checks to ensure all requested validation layers are supported.
pub fn check_validation_layer_support(entry: &Entry) -> VkResult<bool> {
    let mut layer_properties = unsafe {
        entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| CStr::from_ptr(l.layer_name.as_ptr()))
            .collect::<Vec<_>>()
    };

    unsafe {
        for layer in VALIDATION_LAYERS {
            if !layer_properties.contains(&CStr::from_ptr(layer)) {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

pub extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _type_: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void
) -> vk::Bool32 {
    unsafe {
        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            error!("{:?}", &CStr::from_ptr((*callback_data).message));
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            warn!("{:?}", &CStr::from_ptr((*callback_data).message));
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            info!("{:?}", &CStr::from_ptr((*callback_data).message));
        } else {
            debug!("{:?}", &CStr::from_ptr((*callback_data).message));
        }
    }

    vk::FALSE
}

/// Creates a new Vulkan instance, utilizing a reference to a window to use for querying/drawing,
/// and an entry granting access to the Vulkan API.
pub fn create_instance(
    window: &Window,
    entry: &Entry,
    engine_data: &mut VEngineData
) -> Result<Instance, anyhow::Error> {
    // Ensure that all requested validation layers are supported by Vulkan if validation is
    // currently enabled.
    if VALIDATION_ENABLED == true && !check_validation_layer_support(&entry)? {
        return Err(anyhow!("Requested validation layer is not supported!"));
    }

    // Define our debug messenger info, assigning a user callback and declaring the message types
    // and severities we want to catch.
    let mut debug_messenger_info =
        vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            )
            .user_callback(Some(debug_callback));

    // Define some information about the application.
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Test Vulkan Application")
        .application_version(Version::new(1, 0, 0).into())
        .engine_name(b"No Engine")
        .engine_version(Version::new(1, 0, 0).into())
        .api_version(Version::new(1, 0, 0).into());

    // Get the required instance extensions necessary to interface with the window.
    let mut extensions = get_required_instance_extensions(&window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Check if validation is enabled. If it is, we need to add the debug utils extension manually.
    // This allows us to specify a user callback for logging messages.
    let mut extensions = if VALIDATION_ENABLED == true {
        let mut v_extensions = extensions;
        v_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());

        v_extensions
    } else {
        extensions
    };

    // Due to MoltenVK being a third party Vulkan provider, it's necessary to explicitly
    // declare that the programmer understands that the Vulkan implementation being used *may* not
    // conform directly with the Vulkan specification. Here we define the extension for
    // acknowledging the portability conflict.
    #[cfg(target_os = "macos")]
    {
        &extensions.push(
            vk::ExtensionName::from_bytes(b"VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME")
                .as_ptr(),
        );
    }

    let mut instance_create_flags = vk::InstanceCreateFlags::empty();

    // Due to MoltenVK being a third party Vulkan provider, it's necessary to explicitly
    // declare that the programmer understands that the Vulkan implementation being used *may* not
    // conform directly with the Vulkan specification. Here we append the flag for enabling
    // portability to macOS.
    #[cfg(target_os = "macos")]
    {
        instance_create_flags =
            instance_create_flags.union(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }

    // Define some Instance information to use for Vulkan, including the extensions and
    // validation layers to activate, if applicable.
    let mut instance_debug_info = debug_messenger_info.clone();

    let instance_create_info = if VALIDATION_ENABLED == true {
        // Validation is enabled, include the global validation layers.
        vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&VALIDATION_LAYERS)
            .push_next(&mut instance_debug_info)
            .flags(instance_create_flags)
    } else {
        // Validation is not enabled, do not include the validation layers.
        vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&extensions)
            .flags(instance_create_flags)
    };

    // Create an instance.
    let instance = unsafe {
        entry
            .create_instance(&instance_create_info, None)
            .map_err(|e| anyhow!(e))?
    };

    // Create a debug messenger.
    let messenger = unsafe {
        instance.create_debug_utils_messenger_ext(&debug_messenger_info, None)
            .map_err(|e| anyhow!(e))?
    };

    engine_data.debug_messenger = Some(messenger);
    Ok(instance)
}
