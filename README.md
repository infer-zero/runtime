# infer-runtime

This is an interface and runtime for implementing inference models in Zig.

Provide the most common foundation as well as an interface with the minimal necessary for a model to offer good inference.

## Parts

### Data

These are the data parts that every model will need.

- Tensor: Tensor data with with dequantization (`toF32`, `toF16`) and layout calculations. Supports: F32, BF16, F16,Q8_0,Q4_0,Q4_K_M.
- Message: Chat turns (user / assistant / tool_result), tool specs, parameter schemas.
- Vocabulary: Tokens, SpecialTokens, Normalizers and etc.

### Common functions

- Sampler: Token sampling with temperature, top-k, top-p, min-p, repetition penalty.
- Tokenizer: BPE encode/decode.

### Interfaces (factory + session)

Each interface contains a VTable, and some optional methods depending on model capabilities.

- Engine: Minimal model metadata plus a `Context` factory.
- Context: The live conversation: history, logits buffer, sampler, KV-cache position. And Methods to reset context, prefill and get next token/text.

### Aggregate + overlays (also Interfaces)

- Model: view-only aggregate: `{ tokenizer, engine, chat }`.
- Chat (+ nested Tool):optional chat-template overlay: `formatSystem` / `formatMessage` fn pointers, `assistant_prime` / `assistant_prime_no_thinking` / `end_of_turn_suffix` strings, `special_tokens`, `eos_token_id`, and optional nested `Tool` for tool-calling.

### Driver

This where most users of runtime will interact.

- ChatSession: multi-turn chat driver. Borrows a `*Context` + a `Chat` overlay and runs the streaming state machine for thinking blocks, tool calls, and end-of-turn suffix injection.

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

// 1. A variant exposes init/deinit/toModel/createContext.
var variant = try MyVariant.init(allocator, model_path);
defer variant.deinit();

// 2. `toModel()` hands back a lightweight aggregate view.
var model = variant.toModel();

// 3. createContext returns a variant-specific session wrapper;
//    its `.interface` field is the `*runtime.Context` that drivers use.
var ctx = try variant.createContext(allocator, .{
    .sampler_options = .{ .temperature = 0.7, .top_k = 40, .top_p = 0.95, .min_p = 0.05, .repetition_penalty = 1.1, .repetition_penalty_last_n = 64, .seed = null },
    .max_len = null,
});
defer ctx.deinit();
const context: *runtime.Context = &ctx.interface;

// 4. Raw completion: prefill text, loop next(), stop on end-of-turn.
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
const chat = model.chat orelse return error.ModelDoesNotSupportChat;
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
