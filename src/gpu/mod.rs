//! GPU backend (feature `gpu`): Vulkan compute with GLSL kernels that reproduce the CPU sim bit for bit.
//!
//! Parallelism is across races, never within one: each lane of a wave runs one whole race in the CPU's order,
//! with its own `System.Random`, so the result can match [`crate::sim`] seed for seed. What remains is float
//! semantics, handled in three places:
//!
//! * [`spirv`] marks every float op `NoContraction` and requests the device's float-controls modes.
//! * `shaders/lib/exact.glsl` holds the `sim::util` helpers and correctly rounded `/` and `sqrt` (the Vulkan
//!   defaults are approximate).
//! * `gpu_selftest` checks both against the Rust functions on this device.

pub mod context;
mod exact_ref;
pub mod layout;
pub mod pack;
pub mod run;
pub mod shaders;
pub mod skills;
pub mod spirv;

pub use context::{Buffer, DeviceInfo, FloatControls, Gpu, Kernel, as_bytes};
