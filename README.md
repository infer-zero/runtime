# runtime

The inference runtime: the contract variants implement + the session/driver layer that runs any variant satisfying it. Variants depend on `runtime` at a *type* level (they implement its interfaces); applications depend on `runtime` at an *API* level (they drive inference through `Model`, `Context`, `ChatSession`).

## Modules

### Data (leaves)

- **Tensor** — Tensor data with BF16/FP32/FP16/Q8_0/Q4_0/Q4_1/Q6_K support. Includes dequantization (`toF32`, `toF16`) and layout calculations.
- **Message** — Chat turns (user / assistant / tool_result), tool specs, parameter schemas. Lives in `chat_session.zig` alongside `Chat`/`ChatSession` since this is the "chat vocabulary."

### Services

- **Sampler** — Token sampling with temperature, top-k, top-p, min-p, repetition penalty (applied once per unique token in a sliding window).
- **Tokenizer** + nested **Vocabulary** — BPE encode/decode. Vocabulary owns the tokenizer-level `eos_token_id`.

### Interfaces (session + factory)

- **Context** (+ nested `Context.VTable`) — the live conversation: history, logits buffer, sampler, KV-cache position. Variants embed a `Context` as a field on their concrete session struct and fill the 4-method vtable (`restart`, `truncateTo`, `prefill`, `next`); `@fieldParentPtr` recovers the wrapper inside vtable impls (std-library `Reader`/`Writer` pattern).
- **Engine** (+ nested `Engine.VTable`) — per-variant metadata (`vocabulary_size`, `max_len`) plus a single-method factory (`createContext`). Embedded as a field on the concrete variant model type and pointed at by `Model.engine`.

### Aggregate + overlays

- **Model** — view-only aggregate: `{ tokenizer, engine, chat }`. No `deinit` — the caller owns the concrete variant and deinits it directly.
- **Chat** (+ nested **Tool**) — optional chat-template overlay: `formatSystem` / `formatMessage` fn pointers, `assistant_prime` / `assistant_prime_no_thinking` / `end_of_turn_suffix` strings, `special_tokens`, `eos_token_id`, and optional nested `Tool` for tool-calling.

### Driver

- **ChatSession** — multi-turn chat driver. Borrows a `*Context` + a `Chat` overlay and runs the streaming state machine for thinking blocks, tool calls, and end-of-turn suffix injection. Implements **ephemeral thinking**: reasoning tokens live in the KV cache only during their own turn and are rolled back at the turn boundary via `Context.truncateTo`.

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/runtime
```

Then in your `build.zig`:

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
defer session.deinit();  // frees session arena; does NOT deinit context (ChatSession borrows it)

try session.sendText("What's 17 + 23?");
const reply = try session.receive();
```

## Implementation contract for variants

A variant's concrete model struct must:

1. Embed `engine: runtime.Engine` as a field, filled at init time with its `vocabulary_size`, `max_len`, and a pointer to a `const engine_vtable: Engine.VTable` that carries `createContext`.
2. Provide `init(allocator, path) !*Self`, `deinit(self)`, and `toModel(self) runtime.Model`.
3. Expose a variant-specific Context type that:
   - Embeds `interface: runtime.Context` as a field (not named `base` — that collides with `const runtime = @import("runtime")` at file-as-struct scope).
   - Fills a `const context_vtable: Context.VTable` with `restart` / `truncateTo` / `prefill` / `next` impls, each recovering `*Self` via `@fieldParentPtr("interface", ctx)`.
   - Exposes `init(allocator, *Model, Options) !*Self` and `deinit(self)`.
4. Optionally build a `Chat` struct literal in `toModel` with the variant's chat-template fn pointers and static prime strings.

## Dependencies

None.

## License

MIT
