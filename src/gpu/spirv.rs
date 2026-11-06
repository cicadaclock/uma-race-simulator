//! Load-time SPIR-V patching that makes the driver keep the CPU's float semantics.
//!
//! GLSL has no global "no fast math" switch: `precise` covers only the expressions feeding one variable, and a
//! missed site silently becomes a fused multiply-add. So instead of trusting the shader source, every float
//! arithmetic result in the module gets `NoContraction`, and the entry points get the float-controls execution
//! modes the device supports (denormals preserved, signed zero / Inf / NaN preserved, round to nearest even for
//! conversions). Rust never contracts or flushes, so this is what bit-exact means on the GPU side.

/// Float-controls execution modes to request (each only when the device reports support).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FloatModes {
    pub denorm_preserve_32: bool,
    pub denorm_preserve_64: bool,
    pub signed_zero_inf_nan_preserve_32: bool,
    pub signed_zero_inf_nan_preserve_64: bool,
    pub rounding_rte_32: bool,
    pub rounding_rte_64: bool,
}

const MAGIC: u32 = 0x0723_0203;

// Opcodes.
const OP_EXTENSION: u16 = 10;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_EXECUTION_MODE_ID: u16 = 331;
const OP_DECORATE: u16 = 71;
/// Float arithmetic whose result may be contracted: OpFNegate, OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem, OpFMod,
/// OpVectorTimesScalar, OpMatrixTimesScalar, OpVectorTimesMatrix, OpMatrixTimesVector, OpMatrixTimesMatrix,
/// OpOuterProduct, OpDot.
const FLOAT_ARITH: &[u16] = &[127, 129, 131, 133, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148];
/// Preamble sections that come before the annotations: capabilities .. debug info.
const PREAMBLE: &[u16] = &[
    OP_CAPABILITY, OP_EXTENSION, OP_EXT_INST_IMPORT, OP_MEMORY_MODEL, OP_ENTRY_POINT, OP_EXECUTION_MODE,
    OP_EXECUTION_MODE_ID, 2, 3, 4, 5, 6, 7, 330,
];
/// Annotation instructions: OpDecorate .. OpGroupMemberDecorate, OpDecorateId, OpDecorateString(s).
const ANNOTATIONS: &[u16] = &[71, 72, 73, 74, 75, 332, 5632, 5633];

// Operands.
const DECORATION_NO_CONTRACTION: u32 = 42;
const CAP_DENORM_PRESERVE: u32 = 4464;
const CAP_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4466;
const CAP_ROUNDING_MODE_RTE: u32 = 4467;
const MODE_DENORM_PRESERVE: u32 = 4459;
const MODE_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4461;
const MODE_ROUNDING_MODE_RTE: u32 = 4462;

/// What [`patch`] changed, for logging and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatchStats {
    pub no_contraction_added: usize,
    pub execution_modes_added: usize,
}

#[derive(Clone, Copy)]
struct Inst {
    at: usize,
    len: usize,
    op: u16,
}

fn instructions(words: &[u32]) -> Result<Vec<Inst>, String> {
    if words.len() < 5 || words[0] != MAGIC {
        return Err("not a SPIR-V module".into());
    }
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        let op = (words[at] & 0xffff) as u16;
        if len == 0 || at + len > words.len() {
            return Err(format!("malformed instruction at word {at}"));
        }
        out.push(Inst { at, len, op });
        at += len;
    }
    Ok(out)
}

/// Returns the patched module. Idempotent: existing `NoContraction` decorations, capabilities and execution modes
/// are not duplicated.
pub fn patch(words: &[u32], modes: FloatModes) -> Result<(Vec<u32>, PatchStats), String> {
    let insts = instructions(words)?;
    let version = words[1];

    let mut caps = Vec::new();
    let mut entry_points = Vec::new();
    let mut existing_modes = Vec::new();
    let mut decorated = std::collections::HashSet::new();
    let mut float_results = Vec::new();
    let mut has_float_controls_ext = false;
    for i in &insts {
        let w = &words[i.at..i.at + i.len];
        match i.op {
            OP_CAPABILITY => caps.push(w[1]),
            OP_EXTENSION => {
                let bytes: Vec<u8> = w[1..].iter().flat_map(|x| x.to_le_bytes()).collect();
                let name = bytes.split(|&b| b == 0).next().unwrap_or(&[]);
                has_float_controls_ext |= name == b"SPV_KHR_float_controls";
            }
            OP_ENTRY_POINT => entry_points.push(w[2]),
            OP_EXECUTION_MODE if i.len >= 3 => existing_modes.push((w[1], w[2], w.get(3).copied())),
            OP_DECORATE if i.len >= 3 && w[2] == DECORATION_NO_CONTRACTION => {
                decorated.insert(w[1]);
            }
            op if FLOAT_ARITH.contains(&op) && i.len >= 3 => float_results.push(w[2]),
            _ => {}
        }
    }

    // New capabilities and execution modes.
    let wanted: [(bool, u32, u32, u32); 6] = [
        (modes.denorm_preserve_32, CAP_DENORM_PRESERVE, MODE_DENORM_PRESERVE, 32),
        (modes.denorm_preserve_64, CAP_DENORM_PRESERVE, MODE_DENORM_PRESERVE, 64),
        (modes.signed_zero_inf_nan_preserve_32, CAP_SIGNED_ZERO_INF_NAN_PRESERVE, MODE_SIGNED_ZERO_INF_NAN_PRESERVE, 32),
        (modes.signed_zero_inf_nan_preserve_64, CAP_SIGNED_ZERO_INF_NAN_PRESERVE, MODE_SIGNED_ZERO_INF_NAN_PRESERVE, 64),
        (modes.rounding_rte_32, CAP_ROUNDING_MODE_RTE, MODE_ROUNDING_MODE_RTE, 32),
        (modes.rounding_rte_64, CAP_ROUNDING_MODE_RTE, MODE_ROUNDING_MODE_RTE, 64),
    ];
    let mut new_caps: Vec<u32> = Vec::new();
    let mut new_modes: Vec<[u32; 4]> = Vec::new();
    for &(on, cap, mode, width) in &wanted {
        if !on {
            continue;
        }
        if !caps.contains(&cap) && !new_caps.contains(&cap) {
            new_caps.push(cap);
        }
        for &ep in &entry_points {
            if !existing_modes.contains(&(ep, mode, Some(width))) {
                new_modes.push([(4 << 16) | OP_EXECUTION_MODE as u32, ep, mode, width]);
            }
        }
    }
    // Before SPIR-V 1.4 the float controls need their extension.
    let need_ext = !new_caps.is_empty() && version < 0x0001_0400 && !has_float_controls_ext;

    let mut seen = std::collections::HashSet::new();
    let decorations: Vec<u32> = float_results.into_iter().filter(|id| !decorated.contains(id) && seen.insert(*id)).collect();

    // Insertion points: after the last capability, after the last extension (or the capabilities), after the last
    // entry point / execution mode, and at the end of the annotation section.
    let last_of = |ops: &[u16]| insts.iter().filter(|i| ops.contains(&i.op)).map(|i| i.at + i.len).max();
    let after_caps = last_of(&[OP_CAPABILITY]).ok_or("no OpCapability")?;
    let after_ext = last_of(&[OP_EXTENSION]).unwrap_or(after_caps);
    let after_modes = last_of(&[OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_EXECUTION_MODE_ID]).ok_or("no OpEntryPoint")?;
    let annotations_end = insts
        .iter()
        .find(|i| !PREAMBLE.contains(&i.op) && !ANNOTATIONS.contains(&i.op))
        .map(|i| i.at)
        .unwrap_or(words.len());

    let mut out = Vec::with_capacity(words.len() + 4 * decorations.len() + 16);
    let mut cursor = 0;
    let mut emit_until = |out: &mut Vec<u32>, to: usize| {
        out.extend_from_slice(&words[cursor..to]);
        cursor = to;
    };
    emit_until(&mut out, after_caps);
    for &c in &new_caps {
        out.extend_from_slice(&[(2 << 16) | OP_CAPABILITY as u32, c]);
    }
    emit_until(&mut out, after_ext);
    if need_ext {
        let mut s = b"SPV_KHR_float_controls".to_vec();
        s.push(0);
        while s.len() % 4 != 0 {
            s.push(0);
        }
        out.push((((1 + s.len() / 4) as u32) << 16) | OP_EXTENSION as u32);
        out.extend(s.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
    }
    emit_until(&mut out, after_modes);
    for m in &new_modes {
        out.extend_from_slice(m);
    }
    emit_until(&mut out, annotations_end);
    for &id in &decorations {
        out.extend_from_slice(&[(3 << 16) | OP_DECORATE as u32, id, DECORATION_NO_CONTRACTION]);
    }
    emit_until(&mut out, words.len());

    Ok((out, PatchStats { no_contraction_added: decorations.len(), execution_modes_added: new_modes.len() }))
}

/// Result ids of float arithmetic instructions that lack `NoContraction` (empty after [`patch`]).
pub fn uncovered_float_ops(words: &[u32]) -> Result<Vec<u32>, String> {
    let insts = instructions(words)?;
    let mut decorated = std::collections::HashSet::new();
    let mut results = Vec::new();
    for i in &insts {
        let w = &words[i.at..i.at + i.len];
        if i.op == OP_DECORATE && i.len >= 3 && w[2] == DECORATION_NO_CONTRACTION {
            decorated.insert(w[1]);
        } else if FLOAT_ARITH.contains(&i.op) && i.len >= 3 {
            results.push(w[2]);
        }
    }
    Ok(results.into_iter().filter(|id| !decorated.contains(id)).collect())
}

/// SPIR-V bytes (as embedded with `include_bytes!`) to words.
pub fn words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V length not a multiple of 4");
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: FloatModes = FloatModes {
        denorm_preserve_32: true,
        denorm_preserve_64: true,
        signed_zero_inf_nan_preserve_32: true,
        signed_zero_inf_nan_preserve_64: true,
        rounding_rte_32: true,
        rounding_rte_64: true,
    };

    #[test]
    fn covers_every_float_op_and_is_idempotent() {
        for (name, spv) in crate::gpu::shaders::ALL {
            let w = words(spv);
            assert!(!uncovered_float_ops(&w).unwrap().is_empty(), "{name}: expected float ops");
            let (p, stats) = patch(&w, ALL).unwrap();
            assert!(uncovered_float_ops(&p).unwrap().is_empty(), "{name}");
            assert_eq!(stats.execution_modes_added, 6, "{name}");
            let (q, again) = patch(&p, ALL).unwrap();
            assert_eq!(p, q, "{name}: second patch changed the module");
            assert_eq!(again, PatchStats::default());
        }
    }

    #[test]
    fn no_modes_means_decorations_only() {
        let w = words(crate::gpu::shaders::SELFTEST_OPS);
        let (p, stats) = patch(&w, FloatModes::default()).unwrap();
        assert_eq!(stats.execution_modes_added, 0);
        assert_eq!(p.len(), w.len() + 3 * stats.no_contraction_added);
    }
}
