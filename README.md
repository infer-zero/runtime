# base

Core abstractions for LLM inference: tensor types, tokenization, sampling, and a type-erased model interface.

## Modules

- **Tensor** — Tensor data with support for BF16, FP32, FP16, Q8_0, Q4_0, Q4_1, and Q6_K data types. Includes dequantization (`toF32`, `toF16`) and layout calculations.
- **Vocabulary** — BPE vocabulary with encoding/decoding maps and special token support.
- **Tokenizer** — Text-to-token encoding and token-to-text decoding.
- **Sampler** — Token sampling with temperature, top-k, top-p, min-p, and repetition penalty.
- **Model** — Type-erased model interface using a vtable pattern. Wraps any concrete model type behind `prefill()`, `next()`, `vocabulary()`, and `chatFormat()`.
- **Runtime** — High-level inference runtime. Manages a `Context` for stateful inference with `prefill()` and `next()`.
- **Meta** — Model metadata: architecture, format (GGUF/safetensors), quantization, MoE flag.
- **Message** — Chat message structure (role + content).

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/base
```

Then in your `build.zig`:

```zig
const base_dep = b.dependency("infer_base", .{ .target = target, .optimize = optimize });
my_mod.addImport("base", base_dep.module("infer_base"));
```

```zig
const base = @import("base");

// Wrap a concrete model behind the type-erased interface
var model = base.Model.wrap(&my_model);

// Create a runtime with sampling options
var runtime = base.Runtime.init(allocator, &model, .{ .temperature = 0.7 });
```

## Dependencies

None.

## License

MIT
