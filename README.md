# infer-runtime

This is an interface and runtime for implementing inference models in Zig.

Provide the most common foundation as well as an interface with the minimal necessary for a model to offer good inference.

## Parts

### Data

These are the data parts that every model will need.

- Tensor: Tensor data with with dequantization (`toF32`, `toF16`) and layout calculations. Supports: F32, BF16, F16,Q8_0,Q4_0,Q4_K_M.
- Message: Chat turns (user / assistant / tool_result), tool specs, parameter schemas (`message.zig`).
- Vocabulary: Tokens, SpecialTokens, Normalizers and etc. (data for the default BPE tokenizer).

### Interfaces

Every interface follows the same idiom: a file-as-a-struct view type with data fields plus a `vtable`. Implementations embed the interface as a field and recover their concrete type with `@fieldParentPtr` (stateless implementations can share a single `vtable` const instead).

- Model: the polymorphic handle for one loaded variant: `{ tokenizer, chat, sampler_presets }` + `VTable{ createContext, destroy }`. Returned by a family's `open`; owns the variant's lifecycle (`deinit`).
- Context: the live conversation: history, logits buffer, borrowed sampler, KV-cache position. `VTable{ restart, truncateTo, prefill, next, destroy? }` plus methods to prefill and get the next token/text.
- Tokenizer: `eos_token_id` + `VTable{ encode, decode }`. Default implementation: `Tokenizer.Bpe` (BPE over a loader-built `Vocabulary`).
- Sampler: standard `Options` (temperature, top-k, top-p, min-p, repetition penalty) + `VTable{ sample }`. Default implementation: `Sampler.Default`. Presets/overrides speak the standard `Options` regardless of implementation.
- Chat: optional chat-template overlay: `assistant_prime` / `assistant_prime_no_thinking` / `end_of_turn_suffix` strings, `special_tokens`, `eos_token_id` + `VTable{ formatSystem, formatMessage, parseToolCall? }`. Stateful templates (e.g. wrappers over an external template engine) embed it; `runtime.hermes` provides a ready-made Hermes-JSON `parseToolCall`.

### Driver

This where most users of runtime will interact.

- ChatSession: multi-turn chat driver. Borrows a `*Context` + a `*Chat` and runs the streaming state machine for thinking blocks, tool calls, and end-of-turn suffix injection.

## Usage

Fetch the library:

```bash
zig fetch --save git+https://github.com/infer-zero/runtime
```

Add the model in your `build.zig`:

```zig
const runtime_dep = b.dependency("infer_runtime", .{ .target = target, .optimize = optimize });
my_mod.addImport("runtime", runtime_dep.module("infer_runtime"));
```

### Running inference

```zig
const runtime = @import("runtime");

// 1. A family's `open` loads the file, heap-allocates its aggregate,
//    and returns the `*runtime.Model` handle embedded in it.
const model: *runtime.Model = try my_family.open(io, allocator, model_path, .{});
defer model.deinit(); // tears down the whole variant (weights, tokenizer, chat)

// 2. createContext returns a `*runtime.Context` embedded in a
//    variant-specific wrapper; the model's tokenizer and (when none is
//    supplied) an embedded default sampler are wired automatically.
const context = try model.createContext(io, allocator, .{
    .sampler_options = .{ .temperature = 0.7, .top_k = 40, .top_p = 0.95, .min_p = 0.05, .repetition_penalty = 1.1, .repetition_penalty_last_n = 64, .seed = null },
    .max_len = null,
});
defer if (context.vtable.destroy) |destroy| destroy(context);

// 3. Raw completion: prefill text, loop next(), stop on end-of-turn.
try context.prefill("Once upon a time");
while (true) {
    const text = try context.next();
    defer allocator.free(text);
    if (model.isEndOfTurn(context.current_token)) break;
    try std.fs.File.stdout().writeAll(text);
}
```

### Multi-turn chat

```zig
const chat = model.chat orelse return error.ModelDoesNotSupportChat; // *runtime.Chat
var session = try runtime.ChatSession.init(allocator, chat, context, .{
    .system_prompt = "You are a helpful assistant.",
    .thinking = false,
});
defer session.deinit();  // frees session arena; does not deinit context (ChatSession borrows it)

try session.sendText("What's 17 + 23?");
const reply = try session.receive();
```

## AI Usage

- The first full version of this library was hand written. 
- Some functions, fixes and zig version migratation were AI assisted. 
- Comments and docs were AI written and human edited.
- All was human reviewed.
- The design, interfaces and archtecture is my own.

## License

MIT
