//! A minimal Vulkan compute context: one device, one compute queue, storage buffers and compute kernels.
//!
//! Every kernel's SPIR-V goes through [`crate::gpu::spirv::patch`] with the float modes this device supports, so
//! shaders get the CPU's float semantics without relying on `precise` in the GLSL. Calls are synchronous: each
//! dispatch or copy is submitted and waited for.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ash::vk;

use super::spirv::{self, FloatModes, PatchStats};

pub type Result<T> = std::result::Result<T, String>;

fn vkerr(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e:?}")
}

/// What the device reports about float behaviour (`VkPhysicalDeviceFloatControlsProperties`).
#[derive(Clone, Copy, Debug, Default)]
pub struct FloatControls {
    pub denorm_preserve_32: bool,
    pub denorm_preserve_64: bool,
    pub denorm_flush_32: bool,
    pub denorm_flush_64: bool,
    pub signed_zero_inf_nan_preserve_32: bool,
    pub signed_zero_inf_nan_preserve_64: bool,
    pub rounding_rte_32: bool,
    pub rounding_rte_64: bool,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub driver_name: String,
    pub driver_info: String,
    pub subgroup_size: u32,
    pub float_controls: FloatControls,
    pub max_workgroups_x: u32,
    /// Nanoseconds per timestamp tick, and the valid timestamp bits of the queue (0: no timestamps).
    pub timestamp_period: f32,
    pub timestamp_bits: u32,
}

impl DeviceInfo {
    pub fn api_version_string(&self) -> String {
        let v = self.api_version;
        format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
    }
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_lock: Mutex<()>,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub info: DeviceInfo,
    /// The float modes requested in every kernel (the supported subset of everything).
    pub modes: FloatModes,
    /// Whether `fma()` rounds once, per width (Vulkan allows two roundings). Passed to every kernel as the
    /// specialization constants `UMA_FMA32_FUSED` (id 0) and `UMA_FMA64_FUSED` (id 1).
    pub fma32_fused: bool,
    pub fma64_fused: bool,
    /// Compiled pipelines, kept between runs in `cache_path` (the driver ignores data from another device or
    /// driver version).
    pipeline_cache: vk::PipelineCache,
    cache_path: Option<std::path::PathBuf>,
}

struct Opened {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    info: DeviceInfo,
    modes: FloatModes,
}

/// A storage buffer. Host buffers are mapped and read/written directly; device buffers go through a staging copy.
pub struct Buffer<'g> {
    device: ash::Device,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    mapped: Option<*mut u8>,
    _gpu: PhantomData<&'g Gpu>,
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A compute pipeline with `buffers` storage buffers at bindings 0.. and `push_bytes` of push constants.
pub struct Kernel<'g> {
    gpu: &'g Gpu,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    buffers: usize,
    push_bytes: u32,
    pub patch: PatchStats,
}

impl Drop for Kernel<'_> {
    fn drop(&mut self) {
        let d = &self.gpu.device;
        unsafe {
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_shader_module(self.module, None);
        }
    }
}

fn c_str(s: std::result::Result<&CStr, std::ffi::FromBytesUntilNulError>) -> String {
    s.map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
}

impl Gpu {
    /// Opens the device chosen by `UMA_GPU` (an index or part of the name), else the first discrete GPU, else the
    /// first device. Requires Vulkan 1.2 with `shaderFloat64` and `shaderInt64`.
    pub fn new() -> Result<Gpu> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("cannot load the Vulkan loader: {e}"))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"uma_race_sim")
            .engine_name(c"uma_race_sim")
            .api_version(vk::API_VERSION_1_2);
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
            .map_err(vkerr("vkCreateInstance"))?;
        // The entry (the loaded library) must outlive the instance, also on the error path.
        let mut gpu = match Self::open(&instance) {
            Ok(o) => Gpu {
                _entry: entry,
                instance,
                device: o.device,
                queue: o.queue,
                queue_lock: Mutex::new(()),
                command_pool: o.command_pool,
                descriptor_pool: o.descriptor_pool,
                memory: o.memory,
                info: o.info,
                modes: o.modes,
                fma32_fused: false,
                fma64_fused: false,
                pipeline_cache: vk::PipelineCache::null(),
                cache_path: None,
            },
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        // UMA_GPU_PIPELINE_CACHE: the cache file, or "off".
        gpu.cache_path = match std::env::var_os("UMA_GPU_PIPELINE_CACHE") {
            Some(v) if v == "off" => None,
            Some(v) => Some(v.into()),
            None => Some(std::env::temp_dir().join("uma_race_sim_pipelines.bin")),
        };
        let initial = gpu.cache_path.as_ref().and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
        let ci = vk::PipelineCacheCreateInfo::default().initial_data(&initial);
        gpu.pipeline_cache = match unsafe { gpu.device.create_pipeline_cache(&ci, None) } {
            Ok(c) => c,
            // Data the driver rejects: start empty.
            Err(_) => unsafe { gpu.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
                .map_err(vkerr("vkCreatePipelineCache"))?,
        };
        let (f32_fused, f64_fused) = gpu.probe_fma()?;
        gpu.fma32_fused = f32_fused;
        gpu.fma64_fused = f64_fused;
        Ok(gpu)
    }

    /// Writes the pipeline cache to its file (also done when the device closes).
    pub fn save_pipeline_cache(&self) {
        let Some(path) = &self.cache_path else { return };
        let Ok(data) = (unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache) }) else { return };
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    /// Runs fma(a, a, -round(a*a)) for an `a` whose square needs one more bit than the format has: fused, the
    /// result is that bit; rounded twice, it is 0.
    fn probe_fma(&self) -> Result<(bool, bool)> {
        let a32 = 1.0f32 + 1.0 / 4096.0;
        let c32 = -(a32 * a32);
        let a64 = 1.0f64 + 1.0 / 134217728.0;
        let c64 = -(a64 * a64);
        let w = |x: f64| [x.to_bits() as u32, (x.to_bits() >> 32) as u32];
        let v: [[u32; 2]; 8] = [
            [a32.to_bits(), 0],
            [a32.to_bits(), 0],
            [c32.to_bits(), 0],
            w(a64),
            w(a64),
            w(c64),
            [0, 0],
            [0, 0],
        ];
        let buf = self.upload(&v)?;
        let k = self.kernel(super::shaders::PROBE, 1, 0)?;
        k.dispatch(&[&buf], &[], 1)?;
        let out: Vec<[u32; 2]> = self.read(&buf, 8)?;
        let r32 = f32::from_bits(out[6][0]);
        let r64 = f64::from_bits(out[7][0] as u64 | (out[7][1] as u64) << 32);
        Ok((r32 == a32.mul_add(a32, c32) && r32 != 0.0, r64 == a64.mul_add(a64, c64) && r64 != 0.0))
    }

    /// Every device the loader sees, for listing.
    pub fn describe_all() -> Result<Vec<DeviceInfo>> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("cannot load the Vulkan loader: {e}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
            .map_err(vkerr("vkCreateInstance"))?;
        let devices = unsafe { instance.enumerate_physical_devices() }.map_err(vkerr("vkEnumeratePhysicalDevices"));
        let out = devices.map(|ds| ds.iter().enumerate().map(|(i, &pd)| Self::describe(&instance, pd, i)).collect());
        unsafe { instance.destroy_instance(None) };
        out
    }

    fn describe(instance: &ash::Instance, pd: vk::PhysicalDevice, index: usize) -> DeviceInfo {
        let mut fc = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut fc).push_next(&mut sg).push_next(&mut drv);
        unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
        let props = p2.properties;
        let t = |b: vk::Bool32| b == vk::TRUE;
        DeviceInfo {
            index,
            name: c_str(props.device_name_as_c_str()),
            device_type: props.device_type,
            api_version: props.api_version,
            driver_name: c_str(drv.driver_name_as_c_str()),
            driver_info: c_str(drv.driver_info_as_c_str()),
            subgroup_size: sg.subgroup_size,
            max_workgroups_x: props.limits.max_compute_work_group_count[0],
            timestamp_period: props.limits.timestamp_period,
            timestamp_bits: 0,
            float_controls: FloatControls {
                denorm_preserve_32: t(fc.shader_denorm_preserve_float32),
                denorm_preserve_64: t(fc.shader_denorm_preserve_float64),
                denorm_flush_32: t(fc.shader_denorm_flush_to_zero_float32),
                denorm_flush_64: t(fc.shader_denorm_flush_to_zero_float64),
                signed_zero_inf_nan_preserve_32: t(fc.shader_signed_zero_inf_nan_preserve_float32),
                signed_zero_inf_nan_preserve_64: t(fc.shader_signed_zero_inf_nan_preserve_float64),
                rounding_rte_32: t(fc.shader_rounding_mode_rte_float32),
                rounding_rte_64: t(fc.shader_rounding_mode_rte_float64),
            },
        }
    }

    fn open(instance: &ash::Instance) -> Result<Opened> {
        let pds = unsafe { instance.enumerate_physical_devices() }.map_err(vkerr("vkEnumeratePhysicalDevices"))?;
        if pds.is_empty() {
            return Err("no Vulkan devices".into());
        }
        let infos: Vec<DeviceInfo> = pds.iter().enumerate().map(|(i, &pd)| Self::describe(instance, pd, i)).collect();
        let pick = match std::env::var("UMA_GPU") {
            Ok(want) => match want.parse::<usize>() {
                Ok(i) if i < infos.len() => i,
                _ => infos
                    .iter()
                    .position(|d| d.name.to_lowercase().contains(&want.to_lowercase()))
                    .ok_or_else(|| format!("UMA_GPU={want:?} matches no device"))?,
            },
            Err(_) => {
                let rank = |t: vk::PhysicalDeviceType| match t {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 4,
                    _ => 3,
                };
                (0..infos.len()).min_by_key(|&i| (rank(infos[i].device_type), i)).unwrap()
            }
        };
        let pd = pds[pick];
        let info = infos[pick].clone();
        if info.api_version < vk::API_VERSION_1_2 {
            return Err(format!("{} supports Vulkan {}, 1.2 is required", info.name, info.api_version_string()));
        }
        let mut f2 = vk::PhysicalDeviceFeatures2::default();
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        if f2.features.shader_float64 != vk::TRUE || f2.features.shader_int64 != vk::TRUE {
            return Err(format!("{} lacks shaderFloat64 / shaderInt64", info.name));
        }
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let qf = families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).ok_or("no compute queue")? as u32;
        let mut info = info;
        info.timestamp_bits = families[qf as usize].timestamp_valid_bits;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let features = vk::PhysicalDeviceFeatures::default().shader_float64(true).shader_int64(true);
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_features(&features);
        let device = unsafe { instance.create_device(pd, &dci, None) }.map_err(vkerr("vkCreateDevice"))?;
        let queue = unsafe { device.get_device_queue(qf, 0) };
        let destroy = |e: String| {
            unsafe { device.destroy_device(None) };
            e
        };
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| destroy(format!("vkCreateCommandPool: {e:?}")))?;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1024 }];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(128)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| {
            unsafe { device.destroy_command_pool(command_pool, None) };
            destroy(format!("vkCreateDescriptorPool: {e:?}"))
        })?;
        let memory = unsafe { instance.get_physical_device_memory_properties(pd) };
        let fc = info.float_controls;
        let modes = FloatModes {
            denorm_preserve_32: fc.denorm_preserve_32,
            denorm_preserve_64: fc.denorm_preserve_64,
            signed_zero_inf_nan_preserve_32: fc.signed_zero_inf_nan_preserve_32,
            signed_zero_inf_nan_preserve_64: fc.signed_zero_inf_nan_preserve_64,
            rounding_rte_32: fc.rounding_rte_32,
            rounding_rte_64: fc.rounding_rte_64,
        };
        Ok(Opened { device, queue, command_pool, descriptor_pool, memory, info, modes })
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags, prefer: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        let ok = |i: usize| bits & (1 << i) != 0 && types[i].property_flags.contains(want);
        (0..types.len())
            .find(|&i| ok(i) && types[i].property_flags.contains(prefer))
            .or_else(|| (0..types.len()).find(|&i| ok(i)))
            .map(|i| i as u32)
    }

    fn buffer(&self, size: u64, host: bool) -> Result<Buffer<'_>> {
        let size = size.max(16);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let d = &self.device;
        let buffer = unsafe {
            d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
        }
        .map_err(vkerr("vkCreateBuffer"))?;
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let (want, prefer) = if host {
            (vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, vk::MemoryPropertyFlags::HOST_CACHED)
        } else {
            (vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty())
        };
        let Some(ty) = self.memory_type(req.memory_type_bits, want, prefer) else {
            unsafe { d.destroy_buffer(buffer, None) };
            return Err("no suitable memory type".into());
        };
        let memory = match unsafe {
            d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { d.destroy_buffer(buffer, None) };
                return Err(format!("vkAllocateMemory ({size} bytes): {e:?}"));
            }
        };
        let mut b = Buffer { device: d.clone(), buffer, memory, size, mapped: None, _gpu: PhantomData };
        unsafe { d.bind_buffer_memory(buffer, memory, 0) }.map_err(vkerr("vkBindBufferMemory"))?;
        if host {
            let p = unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.map_err(vkerr("vkMapMemory"))?;
            b.mapped = Some(p as *mut u8);
        }
        Ok(b)
    }

    /// A device-local storage buffer of `size` bytes (contents undefined).
    pub fn device_buffer(&self, size: u64) -> Result<Buffer<'_>> {
        self.buffer(size, false)
    }

    /// A host-visible storage buffer (slower for the GPU; convenient for small data).
    pub fn host_buffer(&self, size: u64) -> Result<Buffer<'_>> {
        self.buffer(size, true)
    }

    /// A device-local buffer holding `data`.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer<'_>> {
        let bytes = as_bytes(data);
        let dst = self.device_buffer(bytes.len() as u64)?;
        self.write(&dst, bytes)?;
        Ok(dst)
    }

    /// Writes `bytes` at offset 0 of `dst`.
    pub fn write(&self, dst: &Buffer, bytes: &[u8]) -> Result<()> {
        assert!(bytes.len() as u64 <= dst.size);
        if let Some(p) = dst.mapped {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
            return Ok(());
        }
        let staging = self.host_buffer(bytes.len() as u64)?;
        self.write(&staging, bytes)?;
        self.submit(|cb| unsafe {
            let region = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes.len() as u64 }];
            self.device.cmd_copy_buffer(cb, staging.buffer, dst.buffer, &region);
        })?;
        Ok(())
    }

    /// Reads `count` values of `T` from the start of `src`.
    pub fn read<T: Copy + Default>(&self, src: &Buffer, count: usize) -> Result<Vec<T>> {
        self.read_at(src, 0, count)
    }

    /// `count` elements of `src` from element `at`.
    pub fn read_at<T: Copy + Default>(&self, src: &Buffer, at: usize, count: usize) -> Result<Vec<T>> {
        let size = std::mem::size_of::<T>();
        let (offset, bytes) = (at * size, count * size);
        assert!((offset + bytes) as u64 <= src.size);
        let mut out = vec![T::default(); count];
        if let Some(p) = src.mapped {
            unsafe { std::ptr::copy_nonoverlapping(p.add(offset), out.as_mut_ptr() as *mut u8, bytes) };
            return Ok(out);
        }
        let staging = self.host_buffer(bytes as u64)?;
        self.submit(|cb| unsafe {
            let region = [vk::BufferCopy { src_offset: offset as u64, dst_offset: 0, size: bytes as u64 }];
            self.device.cmd_copy_buffer(cb, src.buffer, staging.buffer, &region);
        })?;
        let p = staging.mapped.unwrap();
        unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr() as *mut u8, bytes) };
        Ok(out)
    }

    /// Records `f` into a fresh command buffer between full memory barriers, submits it and waits.
    fn submit(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<Duration> {
        let d = &self.device;
        let cb = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(vkerr("vkAllocateCommandBuffers"))?[0];
        let barrier = |dst_stage: vk::PipelineStageFlags, dst_access: vk::AccessFlags| unsafe {
            let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(dst_access)];
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, dst_stage, vk::DependencyFlags::empty(), &mb, &[], &[]);
        };
        let result = (|| {
            unsafe {
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            }
            .map_err(vkerr("vkBeginCommandBuffer"))?;
            barrier(vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            f(cb);
            barrier(vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ);
            unsafe { d.end_command_buffer(cb) }.map_err(vkerr("vkEndCommandBuffer"))?;
            let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(vkerr("vkCreateFence"))?;
            let cbs = [cb];
            let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
            let t0 = Instant::now();
            let r = {
                let _guard = self.queue_lock.lock().unwrap();
                unsafe { d.queue_submit(self.queue, &submit, fence) }
                    .map_err(vkerr("vkQueueSubmit"))
                    .and_then(|_| unsafe { d.wait_for_fences(&[fence], true, u64::MAX) }.map_err(vkerr("vkWaitForFences")))
            };
            let dt = t0.elapsed();
            unsafe { d.destroy_fence(fence, None) };
            r.map(|_| dt)
        })();
        unsafe { d.free_command_buffers(self.command_pool, &[cb]) };
        result
    }

    /// Builds a compute kernel from SPIR-V bytes (patched for this device's float modes).
    pub fn kernel(&self, spirv_bytes: &[u8], buffers: usize, push_bytes: u32) -> Result<Kernel<'_>> {
        Ok(self.kernels(&[(spirv_bytes, buffers, push_bytes)])?.pop().unwrap())
    }

    /// Several kernels, compiled on one thread each (a driver compiles a pipeline on the calling thread).
    pub fn kernels(&self, list: &[(&[u8], usize, u32)]) -> Result<Vec<Kernel<'_>>> {
        let d = &self.device;
        let mut parts = Vec::new();
        for &(spirv_bytes, buffers, push_bytes) in list {
            let (words, patch) = spirv::patch(&spirv::words(spirv_bytes), self.modes)?;
            let module = unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }
                .map_err(vkerr("vkCreateShaderModule"))?;
            let bindings: Vec<_> = (0..buffers as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = unsafe { d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }
                .map_err(vkerr("vkCreateDescriptorSetLayout"))?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: push_bytes }];
            let set_layouts = [set_layout];
            let mut lci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if push_bytes > 0 {
                lci = lci.push_constant_ranges(&ranges);
            }
            let layout = unsafe { d.create_pipeline_layout(&lci, None) }.map_err(vkerr("vkCreatePipelineLayout"))?;
            parts.push((module, set_layout, layout, buffers, push_bytes, patch));
        }
        let spec_data: [vk::Bool32; 2] = [self.fma32_fused as vk::Bool32, self.fma64_fused as vk::Bool32];
        let cache = self.pipeline_cache;
        let pipelines: Vec<std::result::Result<vk::Pipeline, String>> = std::thread::scope(|sc| {
            let handles: Vec<_> = parts
                .iter()
                .map(|&(module, _, layout, ..)| {
                    let d = d.clone();
                    let spec_data = spec_data;
                    sc.spawn(move || {
                        let spec_map = [
                            vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 },
                            vk::SpecializationMapEntry { constant_id: 1, offset: 4, size: 4 },
                        ];
                        let spec = vk::SpecializationInfo::default().map_entries(&spec_map).data(as_bytes(&spec_data));
                        let stage = vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main")
                            .specialization_info(&spec);
                        let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
                        unsafe { d.create_compute_pipelines(cache, &ci, None) }
                            .map(|p| p[0])
                            .map_err(|(_, e)| format!("vkCreateComputePipelines: {e:?}"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("pipeline thread panicked".into()))).collect()
        });
        let mut kernels = Vec::new();
        let mut first_err = None;
        for ((module, set_layout, layout, buffers, push_bytes, patch), p) in parts.into_iter().zip(pipelines) {
            match p {
                Ok(pipeline) => kernels.push(Kernel { gpu: self, module, set_layout, layout, pipeline, buffers, push_bytes, patch }),
                Err(e) => {
                    unsafe {
                        d.destroy_pipeline_layout(layout, None);
                        d.destroy_descriptor_set_layout(set_layout, None);
                        d.destroy_shader_module(module, None);
                    }
                    first_err.get_or_insert(e);
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        self.save_pipeline_cache();
        Ok(kernels)
    }
}

impl Kernel<'_> {
    /// Runs `groups` workgroups with `buffers` bound in order and `push` as push constants; returns the
    /// submit-to-completion time.
    pub fn dispatch(&self, buffers: &[&Buffer], push: &[u8], groups: u32) -> Result<Duration> {
        assert_eq!(buffers.len(), self.buffers, "buffer count");
        assert_eq!(push.len() as u32, self.push_bytes, "push constant size");
        assert!(groups <= self.gpu.info.max_workgroups_x, "too many workgroups");
        let g = self.gpu;
        let d = &g.device;
        let set_layouts = [self.set_layout];
        let set = unsafe {
            d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(g.descriptor_pool).set_layouts(&set_layouts))
        }
        .map_err(vkerr("vkAllocateDescriptorSets"))?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffers.iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { d.update_descriptor_sets(&writes, &[]) };
        let r = g.submit(|cb| unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
            if !push.is_empty() {
                d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            d.cmd_dispatch(cb, groups, 1, 1);
        });
        unsafe { d.free_descriptor_sets(g.descriptor_pool, &[set]) }.map_err(vkerr("vkFreeDescriptorSets"))?;
        r
    }
}

impl Gpu {
    /// Records the dispatches of `kernels` in order, `repeats` times, as one submission, with a memory barrier after
    /// each dispatch; returns the submit-to-completion time. Every kernel gets the same buffers, push constants and
    /// workgroup count. Keep a submission well under 2 s on Windows (TDR).
    pub fn dispatch_sequence(&self, kernels: &[&Kernel], buffers: &[&Buffer], push: &[u8], groups: u32, repeats: u32) -> Result<Duration> {
        self.sequence(kernels, buffers, push, groups, repeats, None)
    }

    /// [`Gpu::dispatch_sequence`] with a timestamp after each dispatch; returns each kernel's GPU time summed over
    /// the repeats. Needs timestamps on the queue (`info.timestamp_bits > 0`).
    pub fn profile_sequence(&self, kernels: &[&Kernel], buffers: &[&Buffer], push: &[u8], groups: u32, repeats: u32) -> Result<Vec<Duration>> {
        if self.info.timestamp_bits == 0 {
            return Err("the compute queue has no timestamps".into());
        }
        let count = kernels.len() as u32 * repeats + 1;
        let d = &self.device;
        let pool = unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(count), None) }
            .map_err(vkerr("vkCreateQueryPool"))?;
        let result = (|| {
            self.sequence(kernels, buffers, push, groups, repeats, Some(pool))?;
            let mut ticks = vec![0u64; count as usize];
            unsafe { d.get_query_pool_results(pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) }
                .map_err(vkerr("vkGetQueryPoolResults"))?;
            let mask = if self.info.timestamp_bits >= 64 { u64::MAX } else { (1u64 << self.info.timestamp_bits) - 1 };
            let mut per = vec![Duration::ZERO; kernels.len()];
            for q in 1..count as usize {
                let dt = ticks[q].wrapping_sub(ticks[q - 1]) & mask;
                per[(q - 1) % kernels.len()] += Duration::from_nanos((dt as f64 * self.info.timestamp_period as f64) as u64);
            }
            Ok(per)
        })();
        unsafe { d.destroy_query_pool(pool, None) };
        result
    }

    fn sequence(&self, kernels: &[&Kernel], buffers: &[&Buffer], push: &[u8], groups: u32, repeats: u32, timestamps: Option<vk::QueryPool>) -> Result<Duration> {
        assert!(groups <= self.info.max_workgroups_x, "too many workgroups");
        let d = &self.device;
        let mut sets = Vec::new();
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffers.iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
        let result = (|| {
            for k in kernels {
                assert_eq!(buffers.len(), k.buffers, "buffer count");
                assert_eq!(push.len() as u32, k.push_bytes, "push constant size");
                let set_layouts = [k.set_layout];
                let set = unsafe {
                    d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptor_pool).set_layouts(&set_layouts))
                }
                .map_err(vkerr("vkAllocateDescriptorSets"))?[0];
                sets.push(set);
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(info)
                    })
                    .collect();
                unsafe { d.update_descriptor_sets(&writes, &[]) };
            }
            self.submit(|cb| unsafe {
                let mb = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                let mut q = 0u32;
                if let Some(pool) = timestamps {
                    d.cmd_reset_query_pool(cb, pool, 0, kernels.len() as u32 * repeats + 1);
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, q);
                    q += 1;
                }
                for _ in 0..repeats {
                    for (k, &set) in kernels.iter().zip(&sets) {
                        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, k.pipeline);
                        d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, k.layout, 0, &[set], &[]);
                        if !push.is_empty() {
                            d.cmd_push_constants(cb, k.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
                        }
                        d.cmd_dispatch(cb, groups, 1, 1);
                        d.cmd_pipeline_barrier(
                            cb,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &mb,
                            &[],
                            &[],
                        );
                        if let Some(pool) = timestamps {
                            d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, q);
                            q += 1;
                        }
                    }
                }
            })
        })();
        if !sets.is_empty() {
            unsafe { d.free_descriptor_sets(self.descriptor_pool, &sets) }.map_err(vkerr("vkFreeDescriptorSets"))?;
        }
        result
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.pipeline_cache != vk::PipelineCache::null() {
                self.save_pipeline_cache();
                self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The bytes of a slice of plain data.
pub fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}
