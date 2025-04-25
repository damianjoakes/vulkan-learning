use crate::{REQUIRED_EXTENSIONS};
use crate::init::{create_instance, create_vulkan_entry};
use anyhow::anyhow;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::ops::Deref;
use std::ptr::null;
use log::warn;
use vulkanalia::vk::{DeviceV1_0, ExtDebugUtilsExtension, Handle, HasBuilder, InstanceV1_0, KhrSurfaceExtension, KhrSwapchainExtension, PhysicalDeviceProperties, PhysicalDeviceType, ShaderStageFlags, SurfaceFormatKHR};
use vulkanalia::window::create_surface;
use vulkanalia::{vk, Device, Entry, Instance};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalSize};
use winit::error::OsError;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};
use crate::shaders::{create_shader_module};

/// Specifies the number of layers each image in a swap chain should consist of. This is always 1,
/// unless developing a stereoscopic 3D application.
const IMAGE_LAYERS_COUNT: u32 = 1;

/// Contains indices on the current GPU for each queue family.
pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }

    pub fn indices(&self) -> Vec<u32> {
        let mut i = Vec::new();
        i.push(*(self.graphics_family.as_ref().clone()).unwrap());
        i.push(*(self.present_family.as_ref().clone()).unwrap());

        i
    }
}

#[derive(Debug)]
pub struct SwapChainSupportDetails {
    /// The surface's capabilities (i.e, min/max number of images in swap chain, min/max width and
    /// height of images.
    pub capabilities: vk::SurfaceCapabilitiesKHR,

    /// Surface format support, including the pixel format and color space. See this page for more
    /// details:
    /// https://learn.microsoft.com/en-us/windows/win32/wic/-wic-codec-native-pixel-formats
    ///
    /// Some formats include:
    ///
    /// ```
    ///  SurfaceFormatKHR {
    ///      format: B8G8R8A8_UNORM,
    ///      color_space: SRGB_NONLINEAR,
    ///  },
    ///  SurfaceFormatKHR {
    ///      format: B8G8R8A8_SRGB,
    ///      color_space: SRGB_NONLINEAR,
    ///  },
    ///  SurfaceFormatKHR {
    ///      format: R8G8B8A8_UNORM,
    ///      color_space: SRGB_NONLINEAR,
    ///  },
    ///  SurfaceFormatKHR {
    ///      format: R8G8B8A8_SRGB,
    ///      color_space: SRGB_NONLINEAR,
    ///  },
    ///  SurfaceFormatKHR {
    ///      format: A2B10G10R10_UNORM_PACK32,
    ///      color_space: SRGB_NONLINEAR,
    ///  },
    /// ```
    pub formats: Vec<vk::SurfaceFormatKHR>,

    /// The presentation mode support for the surface. This can include FIFO (v-sync) mode,
    /// immediate mode, FIFO relaxed, and mailbox (a hybrid of the two, where the GPU does not
    /// block if it's running faster than the monitor, but does not cause tearing).
    pub present_modes: Vec<vk::PresentModeKHR>
}

impl SwapChainSupportDetails {
    /// Creates a new `SwapChainsSupportDetails` utilizing an instance, physical device, and
    /// surface to query.
    pub fn query_new(
        instance: &Instance,
        device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR
    ) -> Result<Self, anyhow::Error> {
        // Query device and surface for compatible surface capabilities.
        let capabilities = unsafe {
            instance.get_physical_device_surface_capabilities_khr(*device, *surface)
                .map_err(|e| anyhow!(e))?
        };

        // Query device and surface for compatible surface formats.
        let formats = unsafe {
            instance.get_physical_device_surface_formats_khr(*device, *surface)
                .map_err(|e| anyhow!(e))?
        };

        // Query device and surface for compatible present modes.
        let present_modes = unsafe {
            instance.get_physical_device_surface_present_modes_khr(*device, *surface)
                .map_err(|e| anyhow!(e))?
        };

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }

    pub fn is_adequate(&self) -> bool {
        (self.present_modes.len() > 0) && (self.formats.len() > 0)
    }

    /// Returns the desired format, or None if it does not exist.
    pub fn choose_surface_format(&self) -> Option<SurfaceFormatKHR> {
        for format in &self.formats {
            if format.format == vk::Format::R8G8B8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return Some(format.clone())
            }
        }

        None
    }

    /// Chooses the present mode for the swap chain. Prefers MAILBOX,
    /// but will return FIFO if that is not available.
    pub fn choose_present_mode(&self) -> vk::PresentModeKHR {
        for mode in &self.present_modes {
            if *mode == vk::PresentModeKHR::MAILBOX {
                return mode.clone()
            }
        }

        vk::PresentModeKHR::FIFO
    }

    /// Chooses the swap extent of the engine. This is the resolution of the swap chain images,
    /// and is equivalent to the resolution of the window we're drawing to in *pixels*.
    ///
    /// There arises a problem with displays which have a high pixel density, this being that the
    /// window that we created will typically be in screen coordinates, *not* pixels. A display
    /// with a high pixel density may be instantiated at 800x600 as the window dimensions
    /// (for example), but on a high DPI display, screen coordinates and pixels are usually not
    /// equal.
    ///
    /// Another thing (which is what we'll do here) is that we need to check whether the window is
    /// resizable or not. If it is resizable, then the width and/or height will be equivalent to
    /// `u32::MAX`. In this instance, we need to specify the swap chain extent manually.
    pub fn choose_swap_extent(&self, window: &Window) -> vk::Extent2D {
        use vulkanalia::vk::HasBuilder;

        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent.clone()
        } else {
            let size = window.inner_size();
            let (width, height) = (size.width, size.height);

            // Get the minimum and maximum supported width/height extents.
            let min_extents = self.capabilities.min_image_extent.clone();
            let max_extents = self.capabilities.max_image_extent.clone();
            let (min_width, max_width, min_height, max_height) =
                (min_extents.width, max_extents.width, min_extents.height, max_extents.height);

            // Create an extent where the width is either the current extent, the minimum extent size,
            // or the maximum extent size. This clamps the extent to no greater than the supported
            // size of the window.
            let extent = vk::Extent2D::builder()
                .width(u32::clamp(width, min_width, max_width))
                .height(u32::clamp(height, min_height, max_height))
                .build();

            extent
        }
    }

    pub fn min_image_count(&self) -> u32 {
        self.capabilities.min_image_count
    }

    pub fn recommended_image_count(&self) -> u32 {
        self.capabilities.min_image_count + 1
    }

    pub fn has_maximum_image_count(&self) -> bool {
        self.capabilities.max_image_count > 0
    }

    /// Returns a clamped 32-bit integer somewhere between the maximum and minimum image count,
    /// or the desired image count if it fits within those bounds. If there is no maximum, returns
    /// the desired image count that was provided, or a minimum of the minimum image count.
    pub fn select_image_count(&self, count: u32) -> u32 {
        if self.has_maximum_image_count() {
            count.clamp(self.capabilities.min_image_count, self.capabilities.max_image_count)
        } else {
            count.clamp(self.capabilities.min_image_count, u32::MAX)
        }
    }
}

pub struct SwapData {
    extent: vk::Extent2D,
    format: vk::Format,
    images: Vec<vk::Image>
}

impl SwapData {
    pub fn new(extent: vk::Extent2D, format: vk::Format, images: Vec<vk::Image>) -> Self {
        Self { extent, format, images }
    }
}

pub struct PipelinePair(vk::ShaderModule, vk::PipelineShaderStageCreateInfo);

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
    pub swap_chain: vk::SwapchainKHR,
    pub swap_data: Option<SwapData>,
    pub image_views: Vec<vk::ImageView>,
    pub shader_modules: Vec<PipelinePair>,
    pub pipeline_layout: vk::PipelineLayout
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
            swap_chain: vk::SwapchainKHR::null(),
            swap_data: None,
            image_views: Vec::new(),
            shader_modules: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null()
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

    /// Creates a graphics pipeline for rendering images to the screen.
    fn create_graphics_pipeline(&mut self) -> Result<(), anyhow::Error> {
        let vert_shader = include_bytes!("../compiled_shaders/tri-vert.spv");
        let frag_shader = include_bytes!("../compiled_shaders/tri-frag.spv");

        let device = self.logical_device.as_ref().unwrap();

        let vert_module = create_shader_module(
            device,
            vert_shader
        )?;

        let frag_module = create_shader_module(
            device,
            frag_shader
        )?;

        // There's an optional member here that we don't use that's worth mentioning,
        // `pSpecializationInfo`. This is used to pass constants to shaders at pipeline creation.
        // This allows reuse of the shader code, but with a different value, to change its
        // functionality. This is more efficient than creating different shaders.
        let vert_stage_pipeline_info =
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(b"main")
                .build();

        let frag_stage_pipeline_info =
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(ShaderStageFlags::FRAGMENT)
                .module(vert_module)
                .name(b"main")
                .build();

        self.shader_modules.push(PipelinePair(vert_module, vert_stage_pipeline_info));
        self.shader_modules.push(PipelinePair(frag_module, frag_stage_pipeline_info));

        // This describes the format of the vertex data that will be passed into the vertex shader.
        // Because (as of right now), we're drawing the triangle from within the shader itself,
        // we're going to fill this out as blank.
        //
        // https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/02_Graphics_pipeline_basics/02_Fixed_functions.html#_vertex_input
        let pipeline_vertex_input_state_info =
            vk::PipelineVertexInputStateCreateInfo::builder().build();


        // This describes the kind of geometry that is going to be drawn by vertices, and whether
        // primitive restart should be enabled. Primitive restart is a technique for drawing
        // primitives that allows the drawing call to create multiple primitives without repeating
        // draw calls. It's possible to boost efficiency of the application doing this, but we will
        // not be doing this here.
        //
        // https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/02_Graphics_pipeline_basics/02_Fixed_functions.html#_input_assembly
        let pipeline_input_assembly_state_info =
        vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // A viewport describes the region of the framebuffer that the output will be rendered to.
        // The depth values of [0.0f32, 1.0f32] are standard.
        //
        // Viewports define the transformation of an image onto the framebuffer, i.e. the
        // "renderable area", but a scissor rectangle must be created if we want to "cut off"
        // sections of the rendered image.
        //
        // https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/02_Graphics_pipeline_basics/02_Fixed_functions.html#_viewports_and_scissors
        let viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(self.swap_data.as_ref().unwrap().extent.width as f32)
            .height(self.swap_data.as_ref().unwrap().extent.height as f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32);

        let scissor_offset = vk::Offset2D::builder()
            .x(0)
            .y(0);

        let scissors = vk::Rect2D::builder()
            .extent(self.swap_data.as_ref().unwrap().extent)
            .offset(scissor_offset);

        // Though most state information within a render pipeline is immutable, some of it can
        // be changed without recreating the pipeline. This includes the viewport size and the
        // scissor rectangle.
        let mut states = Vec::new();
        states.push(vk::DynamicState::VIEWPORT);
        states.push(vk::DynamicState::SCISSOR);

        let dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::builder()
                .dynamic_states(&states);

        let pipeline_viewport_state_info =
            vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .viewports(&[viewport])
                .scissor_count(1)
                .scissors(&[scissors]);

        // We'll now create the rasterizer. this takes the vertices created by the vertex shader and
        // turns them into pixels (fragments) to be colored by the fragment shader.
        //
        // It also performs depth testing, face culling, and the scissor test, and can be configured
        // to output entire fragments that the fragment shader will color, or just edges of the
        // vertices (known as wireframe rendering).
        let pipeline_rasterization_state_info =
            vk::PipelineRasterizationStateCreateInfo::builder()
                // If true, then fragments that are beyond the depth of the viewport will be clamped
                // to the depth of the viewport, rather than discarded. This requires enabling a GPU
                // feature.
                .depth_clamp_enable(false)
                // If this is set to true, then geometry will never be output by the rasterization
                // stage. This disables all output to the framebuffer.
                .rasterizer_discard_enable(false)
                // Determines how fragments are generated for geometry.
                .polygon_mode(vk::PolygonMode::FILL)
                // Describes the thickness of lines in terms of number of fragments. The maximum
                // values allowed here is dependent on the GPU and anything larger than 1.0f32
                // requires the `wideLines` GPU feature.
                .line_width(1.0f32)
                // The cullMode variable determines the type of face culling to use. You can disable
                // culling, cull the front faces, cull the back faces or both.
                .cull_mode(vk::CullModeFlags::BACK)
                // The frontFace variable specifies the vertex order for faces to be considered
                // front-facing and can be clockwise or counterclockwise.
                .front_face(vk::FrontFace::CLOCKWISE)
                // Alters the depth values of the rasterized image by adding a constant value or
                // biasing them based on a fragment's slope. This can be used for shadow mapping.
                .depth_bias_enable(false);


        // Now, it's time to set up multisampling. Multisampling is a method for anti-aliasing
        // which works by combining the fragment shader results of multiple polygons that rasterize
        // to the same size pixel. Multisampling requires enabling a GPU feature, so for now we're
        // going to keep it disabled.
        //
        // The tutorial states that we'll revisit multisampling a later chapter.
        let pipeline_multisampling_info =
            vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::_1)
                .min_sample_shading(1.0f32)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false);

        // Now, we're going to set up color blending for the fragment shader. Color blending is a
        // transformation done to combine the output color from the fragment shader with the color
        // that's currently in the framebuffer.
        //
        // This can be done one of two ways, either by mixing the old and new value to produce a
        // final color, or to use a bitwise operation to combine the two colors.
        //
        // If `blendEnable` is set to false, then the new color from the fragment shader is provided
        // unmodified.
        //
        // Two structs are required for this configuration.
        let color_blend_attachment_state =
            vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = vec![color_blend_attachment_state];

        let pipeline_color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachment_count(1)
                .attachments(&attachments);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder();

        let pipeline_layout = unsafe {
            self.logical_device.as_ref()
                .unwrap()
                .create_pipeline_layout(
                    &pipeline_layout_info,
                    None
                )
                .map_err(|e| anyhow!(e))?
        };

        self.pipeline_layout = pipeline_layout;

        Ok(())
    }

    /// Creates image views for every image we were given upon creation of the swap chain.
    fn create_image_views(&mut self) -> Result<(), anyhow::Error> {
        use vk::HasBuilder;

        let logical_device = if let Some(d) = self.logical_device.as_ref() {
            d
        } else {
            return Err(anyhow!("Device not initialized. Unable to create image views."));
        };

        let (images, format) = if let Some(swap_data) = self.swap_data.as_ref() {
            (&swap_data.images, &swap_data.format)
        } else {
            return Err(anyhow!("Swap data has not yet been initialized."));
        };

        self.image_views = Vec::with_capacity(images.len());

        for image in images {
            // Allows us to reroute (swizzle) color channels in our shaders.
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            // Describes the image's purpose. Mostly pertains to mipmap levels and image layers.
            // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageSubresourceRange.html
            // It isn't useful to specify the mipmap levels here, as we're not generating mipmaps for
            // the textures that will be rendered to the images provided here (yet?). Instead, we focus
            // on the `aspect_mask` property. This is set to `VK_IMAGE_ASPECT_COLOR_BIT`, as when
            // our format was chosen for color (direct) output to this swap chain, Vulkan requires
            // that this bit be set to the `VK_IMAGE_ASPECT_COLOR_BIT` bit.
            //
            // If we were creating a stereoscopic 3D application, then we would create multiple
            // image views with this information, and the selected layer would correspond to the
            // eye we would want to be rendering for.
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::_2D) // Specifies how the view should be treated.
                .format(*format)
                .components(components)
                .subresource_range(subresource_range);

            let image_view = unsafe {
                logical_device.create_image_view(&create_info, None)
                    .map_err(|e| anyhow!(e))?
            };

            self.image_views.push(image_view);
        }

        Ok(())
    }

    /// Creates a swap chain for the engine's surface.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/01_Presentation/01_Swap_chain.html
    fn create_swap_chain(&mut self) -> Result<(), anyhow::Error> {
        use vulkanalia::vk::HasBuilder;

        let instance = if let Some(instance) = self.instance.as_ref() {
            instance
        } else {
            return Err(anyhow!("Instance not initialized. Unable to create a swap chain."));
        };

        let logical_device = if let Some(d) = self.logical_device.as_ref() {
            d
        } else {
            return Err(anyhow!("Logical device not initialized. Unable to create a swap chain."));
        };

        let swap_chain_support = SwapChainSupportDetails::query_new(
            instance,
            &self.physical_device,
            &self.surface
        )?;

        // Select the desired surface format.
        let surface_format = swap_chain_support.choose_surface_format()
            .ok_or(anyhow!("Unable to select a surface format for the swap chain."))?;

        // Select the desired present mode (currently mailbox).
        let present_mode = swap_chain_support.choose_present_mode();

        // Get a reference to the engine's window.
        let window_ref = self.window.as_ref()
            .ok_or(anyhow!("Window not initialized, can't create swap chain"))?;

        // Create an extent for the swap chain, supporting resizable windows.
        let swap_extent = swap_chain_support.choose_swap_extent(
            window_ref
        );

        let image_count = swap_chain_support.select_image_count(
            swap_chain_support.recommended_image_count()
        );

        // Create the SwapChainCreateInfo. The `image_usage` flag specifies that the images that are
        // prepared by this swap chain will be expected to support a specific rendering method.
        // `vk::ImageUsageFlags::COLOR_ATTACHMENT` specifies that the images that are created by
        // this swap chain will be rendered to directly, without any extra functionality, whereas
        // something like `vk::ImageUsageFlags::TRANSFER_DST` would be used to specify that the
        // images provided by this swap chain must support a memory copy.
        let mut swap_chain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(swap_extent)
            .image_array_layers(IMAGE_LAYERS_COUNT)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .present_mode(present_mode)
            // Specify that a certain transform should be applied to images created by the swap chain if
            // it is supported. We specify the current transformation, as we don't want to transform the
            // images.
            .pre_transform(swap_chain_support.capabilities.current_transform.clone())
            // Specifies if the alpha channel should be used for blending other windows in the
            // window system.
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(true)
            .old_swapchain(vk::Handle::null());


        let queue_family_indices = self.find_queue_families(&self.physical_device)?;
        let indices = queue_family_indices.indices();

        if queue_family_indices.graphics_family.as_ref().unwrap()
            != queue_family_indices.present_family.as_ref().unwrap() {

            swap_chain_create_info = swap_chain_create_info
                // Allows multiple different queues to access images.
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&indices);
        } else {
            swap_chain_create_info = swap_chain_create_info
                // Only one queue can modify an image, explicit transfer required. Best performance.
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let swap_chain = unsafe {
            logical_device.create_swapchain_khr(&swap_chain_create_info, None)
                .map_err(|e| anyhow!(e))?
        };

        self.swap_chain = swap_chain;

        let swap_images = unsafe {
            logical_device.get_swapchain_images_khr(self.swap_chain)
                .map_err(|e| anyhow!(e))?
        };

        self.swap_data = Some(
            SwapData::new(swap_extent, surface_format.format, swap_images)
        );

        Ok(())
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

        // Get our required extension names as c_char pointers to pass to the device create
        // info.
        let required_extensions = REQUIRED_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        // Creates a queue info builder for the graphics queue.
        let device_create_info = vk::DeviceCreateInfo::builder()
            .enabled_features(&features)
            .enabled_extension_names(&required_extensions)
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
    /// Equivalent to the `isDeviceSuitable` function in the docs.
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
                        // Query the device's swap chain support against the current engine's surface.
                        let swap_chain_support = SwapChainSupportDetails::query_new(
                            self.instance.as_ref().unwrap(),
                            &device,
                            &self.surface
                        )?;

                        dbg!(&swap_chain_support);

                        // Find queue family indices.
                        let indices = self.find_queue_families(&device)?;

                        // Check multiple criteria for the current device.
                        if indices.is_complete()
                            && self.check_device_extension_support(&device)?
                            && swap_chain_support.is_adequate()
                        {
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

    /// Queries the physical device for support for the required extensions for the engine.
    ///
    /// https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/01_Presentation/01_Swap_chain.html
    fn check_device_extension_support(&mut self, device: &vk::PhysicalDevice) -> Result<bool, anyhow::Error> {
        if let Some(instance) = self.instance.as_ref() {
            let extension_properties = unsafe {
                instance.enumerate_device_extension_properties(*device, None)
                    .map_err(|e| anyhow!(e))?
                    .iter()
                    .map(|e| e.extension_name)
                    .collect::<Vec<_>>()
            };

            let mut supported = true;
            for ext in REQUIRED_EXTENSIONS {
                if !extension_properties.contains(&ext) {
                    warn!("Required extension {:?} not available!", ext);
                    supported = false;
                }
            }

            Ok(supported)
        } else {
            Err(anyhow!("Instance not initialized. Unable to query physical device extension support."))
        }
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

        // Create a swap chain.
        self.create_swap_chain()?;
        self.create_image_views()?;

        self.create_graphics_pipeline()?;

        Ok(())
    }

    /// Cleans up all open Vulkan device handles.
    fn cleanup(&mut self, event_loop: &ActiveEventLoop) {
        // Close out the event loop.
        event_loop.exit();

        unsafe {
            let instance = self.instance.as_ref().unwrap();

            // Destroy our logical device handle.
            if self.logical_device.is_some() {
                let l_device = self.logical_device.as_ref().unwrap();
                l_device.destroy_swapchain_khr(self.swap_chain, None);

                for view in &self.image_views {
                    l_device.destroy_image_view(*view, None);
                }

                for module in &self.shader_modules {
                    l_device.destroy_shader_module(module.0, None);
                }

                if !self.pipeline_layout.is_null() {
                    l_device.destroy_pipeline_layout(self.pipeline_layout, None);
                }

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
            .with_inner_size(PhysicalSize::new(800, 600));

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
