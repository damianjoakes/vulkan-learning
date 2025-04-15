use crate::init::{create_instance, create_vulkan_entry};
use anyhow::anyhow;
use std::collections::BTreeMap;
use vulkanalia::vk::{
    DeviceV1_0, ExtDebugUtilsExtension, Handle, InstanceV1_0, KhrSurfaceExtension
    , PhysicalDeviceProperties, PhysicalDeviceType,
};
use vulkanalia::window::create_surface;
use vulkanalia::{vk, Device, Entry, Instance};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::error::OsError;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

///
pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

/// A struct containing some engine data for holding utilities, etc.
pub struct VEngineData {
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

/// The primary engine for running the Vulkan instance, managing window creation, and handling
/// system events.
pub struct VulkanEngine {
    pub should_continue: bool,
    pub entry: Entry,
    pub instance: Option<Instance>,
    pub window: Option<Window>,
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: Option<Device>,
    pub graphics_queue: vk::Queue,
    pub presentation_queue: vk::Queue,
    pub graphics_queue_index_count: u32,
    pub present_queue_index_count: u32,
    pub data: VEngineData,
    pub surface: vk::SurfaceKHR,
}

impl VulkanEngine {
    pub fn new() -> Self {
        use vk::Handle;

        Self {
            data: VEngineData {
                debug_messenger: None,
            },
            should_continue: false,
            entry: unsafe { create_vulkan_entry() },
            instance: None,
            window: None,
            graphics_queue_index_count: 0,
            present_queue_index_count: 0,
            logical_device: None,
            physical_device: vk::PhysicalDevice::null(),
            graphics_queue: vk::Queue::null(),
            presentation_queue: vk::Queue::null(),
            surface: vk::SurfaceKHR::null(),
        }
    }

    /// Runs the Vulkan engine, handling all window events and redraw requests.
    pub fn run(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.cleanup(event_loop);
            }

            WindowEvent::RedrawRequested => {
                dbg!(&window_id);
            }

            _ => {}
        }
    }

    /// Creates a new logical device utilizing the physical device selected by the engine.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/04_Logical_device_and_queues.html
    fn create_logical_device(&mut self) -> Result<(), anyhow::Error> {
        use vk::HasBuilder;

        if self.instance.is_none() {
            return Err(anyhow!(
                "Instance not initialized. Unable to create a logical device."
            ));
        }
        let instance = self.instance.as_ref().unwrap();

        let features = vk::PhysicalDeviceFeatures::builder();

        // Specify our device queue information.
        let indices = self.find_queue_families(&self.physical_device)?;

        let mut device_queue_create_infos = Vec::new();

        // Create the graphics pipeline. This one is a priority queue, and should always be
        // instantiated.
        if let Some(graphics) = indices.graphics_family {
            // Create the queue info for the graphics queue.
            let gqci = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics)
                .queue_priorities(&[1.0f32]); // Allows for assigning priorities to different queues.

            device_queue_create_infos.push(gqci);
        }

        // Create the present family. If the graphics family has the same queue family
        // index as the present family, then this present family won't get added, since the
        // two queues families are part of the same queue.
        if let Some(present) = indices.present_family {
            if device_queue_create_infos.get(present as usize).is_none() {
                // Create the queue info for the presentation queue.
                let pqci = vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(indices.present_family.unwrap())
                    .queue_priorities(&[1.0f32]);

                device_queue_create_infos.push(pqci);
            }
        }

        // Creates a queue info builder for the graphics queue.
        let device_create_info = vk::DeviceCreateInfo::builder()
            .enabled_features(&features)
            .queue_create_infos(&device_queue_create_infos);

        // Attempt to get a logical `vk::Device` handle to the physical graphics device.
        let logical_device = unsafe {
            instance
                .create_device(self.physical_device, &device_create_info, None)
                .map_err(|e| anyhow!(e))?
        };

        self.logical_device = Some(logical_device);

        // Check to make sure the logical device is initialized (1000% should be), and get our
        // required queues.
        unsafe {
            let l_device = self.logical_device.as_ref();

            if l_device.is_some() {
                let l_device = l_device.unwrap();

                let graphics_queue = l_device.get_device_queue(
                    indices.graphics_family.unwrap(),
                    self.graphics_queue_index_count,
                );

                let presentation_queue = l_device.get_device_queue(
                    indices.present_family.unwrap(),
                    self.present_queue_index_count,
                );

                self.graphics_queue_index_count += 1;
                self.present_queue_index_count += 1;

                self.graphics_queue = graphics_queue;
                self.presentation_queue = presentation_queue;
            } else {
                self.graphics_queue = vk::Queue::null();
                self.presentation_queue = vk::Queue::null();
            }
        };

        if self.graphics_queue.is_null() {
            return Err(anyhow!("Logical device ended up as `null`!"));
        }

        Ok(())
    }

    /// Selects a highest-ranked physical device on the system for use from Vulkan.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/03_Physical_devices_and_queue_families.html
    fn pick_physical_device(&mut self) -> Result<(), anyhow::Error> {
        // Make sure our instance is still valid.
        if self.instance.is_some() {
            let instance = self.instance.as_ref().unwrap();
            let devices = unsafe {
                instance
                    .enumerate_physical_devices()
                    .map_err(|e| anyhow!(e))?
            };

            let mut dev_map = BTreeMap::new();

            // Iterate all physical devices. Each one will be scored, and, as long as the score
            // surpasses 0 and the device contains a queue family which supports graphics rendering,
            // will be pushed to the device map.
            for device in devices {
                if let Ok(score) = self.get_device_score(&device) {
                    if score > 0 {
                        let indices = self.find_queue_families(&device)?;
                        if indices.graphics_family.is_some() {
                            dev_map.insert(score, device);
                        }
                    }
                }
            }

            // Check to make sure a suitable device was found. The last device in the list is used.
            // This is the device which has the highest score.
            let suitable_device = dev_map.iter().last();
            if suitable_device.is_none() {
                return Err(anyhow!("No suitable devices found!"));
            }

            self.physical_device = *suitable_device.unwrap().1;
        } else {
            return Err(anyhow!(
                "Instance not initialized. Unable to check for physical devices."
            ));
        }

        Ok(())
    }

    /// Checks suitability of all Vulkan-capable devices on the client.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/03_Physical_devices_and_queue_families.html
    fn get_device_score(&self, device: &vk::PhysicalDevice) -> Result<u32, anyhow::Error> {
        // Ensure the Vulkan instance is still initialized.
        if self.instance.is_none() {
            return Err(anyhow!(
                "Instance not initialized. Unable to query physical device for properties."
            ));
        }

        let instance = self.instance.as_ref().unwrap();
        let properties = unsafe { instance.get_physical_device_properties(*device) };

        let features = unsafe { instance.get_physical_device_features(*device) };

        let score = self.score_physical_device(&properties, &features);

        Ok(score)
    }

    /// Accepts a physical device's properties and features, and returns a number which contains the
    /// device's score based on necessary application requirements.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/03_Physical_devices_and_queue_families.html
    fn score_physical_device(
        &self,
        p: &PhysicalDeviceProperties,
        f: &vk::PhysicalDeviceFeatures,
    ) -> u32 {
        // If the geometry shader feature is not supported, return 0.
        if f.geometry_shader == vk::FALSE {
            0
        } else {
            // Rank the device by its type, and it maximum supported texture size.
            let mut score = 0;

            if p.device_type == PhysicalDeviceType::DISCRETE_GPU {
                score += 1000;
            }

            score += p.limits.max_image_dimension_2d;

            score
        }
    }

    /// Locates the indices of each desired queue family on the provided physical device. Returns
    /// an object containing `Some(index)` if the family was provided, and `None` for a specified
    /// family that does not exist.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/03_Physical_devices_and_queue_families.html
    fn find_queue_families(
        &self,
        device: &vk::PhysicalDevice,
    ) -> Result<QueueFamilyIndices, anyhow::Error> {
        if self.instance.is_none() {
            return Err(anyhow!(
                "Instance is not initialized. Unable to find device queue families."
            ));
        }

        let instance = self.instance.as_ref().unwrap();

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        let mut i = 0;
        let mut indices = QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        };

        // Iterate through the physical device's queue families. For each queue family flag we find
        // that we need, set the `QueueFamilyIndices` family to the index of the queue family on the
        // physical device.
        for p in queue_family_properties {
            if vk::QueueFlags::contains(&p.queue_flags, vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }

            // Check to ensure the device contains support for window surfaces.
            let surface_support = unsafe {
                instance
                    .get_physical_device_surface_support_khr(*device, i, self.surface)
                    .map_err(|e| anyhow!(e))?
            };

            if surface_support == true {
                indices.present_family = Some(i);
            }

            if indices.is_complete() {
                break;
            }

            i += 1;
        }

        Ok(indices)
    }

    /// Initializes a platform-agnostic window surface for Vulkan. Vulkanalia can handle this
    /// creation directly, utilizing the `create_surface` function. This can also be done for
    /// platform-specific implementations, utilizing the other instance create surface functions.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/01_Presentation/00_Window_surface.html
    fn init_window_surface(&mut self) -> Result<(), anyhow::Error> {
        let window = self.window.as_ref();
        let window = if let Some(window) = window {
            window
        } else {
            return Err(anyhow!("No window exists to create a surface for."));
        };

        if let Some(instance) = self.instance.as_ref() {
            // Platform-agnostic window surface creation.
            let surface_khr =
                unsafe { create_surface(instance, window, window).map_err(|e| anyhow!(e))? };

            self.surface = surface_khr;

            Ok(())
        } else {
            Err(anyhow!(
                "Instance has not been initialized. Unable to initialize a window surface."
            ))
        }
    }

    /// Initializes Vulkan, creating an instance.
    fn init_vulkan(&mut self) -> Result<(), anyhow::Error> {
        // Make sure `self.window` has been initialized. Vulkan cannot be initialized without
        // a window.
        let window_ref = self.window.as_ref().ok_or(anyhow!(
            "Can't create an instance without first having a valid window!"
        ))?;

        // Attempt to create an instance.
        let instance =
            create_instance(window_ref, &self.entry, &mut self.data).map_err(|e| anyhow!(e))?;

        // Take ownership of the instance within the engine.
        self.instance = Some(instance);

        self.init_window_surface()?;

        // Get a handle to a suitable physical device.
        self.pick_physical_device()?;
        self.create_logical_device()?;

        Ok(())
    }

    /// Cleans up all open Vulkan device handles.
    fn cleanup(&mut self, event_loop: &ActiveEventLoop) {
        // Close out the event loop.
        event_loop.exit();

        // Drop the window handle.
        self.window = None;

        unsafe {
            let instance = self.instance.as_ref().unwrap();

            // Destroy our logical device handle.
            if self.logical_device.is_some() {
                let l_device = self.logical_device.as_ref().unwrap();
                l_device.destroy_device(None);
            }

            // Destroy the attached window surface.
            instance.destroy_surface_khr(self.surface, None);

            // Destroy the internal debug messenger structure.
            instance.destroy_debug_utils_messenger_ext(self.data.debug_messenger.unwrap(), None);
        }

        // Destroy the Vulkan instance.
        unsafe {
            self.instance.as_ref().unwrap().destroy_instance(None);
        }

        // Internally flag that the engine is no longer accepting calls.
        self.should_continue = false;
    }

    /// Initializes a window to use when drawing with Vulkan.
    fn init_window(&mut self, event_loop: &ActiveEventLoop) -> Result<(), OsError> {
        // Use some default window attributes.
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Test Application")
            .with_inner_size(LogicalSize::new(800, 600));

        // Use the event loop to initialize a window.
        let window = event_loop.create_window(window_attributes)?;

        // Hold onto the initialized window.
        self.window = Some(window);
        self.should_continue = true;

        Ok(())
    }
}

impl ApplicationHandler for VulkanEngine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.init_window(event_loop).unwrap();
        }

        self.init_vulkan().unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        &self.run(event_loop, window_id, event);
    }
}
