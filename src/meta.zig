architecture: []const u8,
format: Format,
quantization: Quantization,
is_moe: bool,
name: [:0]const u8,

pub const Format = enum { gguf, safetensors };
pub const Quantization = enum { BF16, Q4_0, Q8_0 };
