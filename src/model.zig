//! The polymorphic handle for one loaded variant. Like `Context`,
//! `Tokenizer`, and `Sampler`, this is a view type embedded inside the
//! family's concrete aggregate (e.g. a heap-allocated `Family`/`Loaded`
//! struct that owns the weights, vocabulary, and chat overlay); the
//! vtable methods recover the owner via `@fieldParentPtr("model", m)`.
//!
//! Lifecycle: the family's opener heap-allocates the aggregate and
//! returns `*Model`. Polymorphic callers (the harness runners) tear the
//! whole variant down with `deinit`; comptime-aware callers that
//! constructed the concrete aggregate themselves may instead call its
//! own deinit directly. Contexts created via `createContext` borrow
//! `tokenizer` and must not outlive the Model.

/// The variant's `Tokenizer` (usually a `Tokenizer.Bpe` owned by the
/// family; external-library wrappers bring their own implementation).
tokenizer: *Tokenizer,
/// Optional chat-template overlay, borrowed from the family aggregate
/// (stateful implementations embed the `Chat` and point here at its
/// interface field). Holder only; `ChatSession` is the actual consumer.
/// Null for completion-only models.
chat: ?*Chat = null,
/// Per-family preset table keyed by `Sampler.Profile`. Borrowed —
/// typically a `pub const` in the family's `common/sampler_defaults.zig`.
/// Null when the family has not wired presets.
sampler_presets: ?*const Sampler.Presets = null,
vtable: *const VTable,

const Model = @This();

pub const VTable = struct {
    /// Heap-allocate a `ConcreteContext`, populate its embedded
    /// `Context` (including the `Context.VTable` pointer), and return
    /// `&concrete.interface`. The caller owns the returned pointer; see
    /// `Model.createContext` for the ownership contract.
    createContext: *const fn (
        *Model,
        std.Io,
        std.mem.Allocator,
        Context.Options,
    ) anyerror!*Context,

    /// Tear down the variant's full allocation — weights, caches,
    /// thread pool, and family aggregate. Polymorphic callers that own
    /// the Model's lifetime call this on shutdown via `Model.deinit`.
    destroy: *const fn (*Model) void,
};

/// Create a live `Context` for a new conversation. The context uses
/// `self.tokenizer` for encode/decode. Returns a `*Context` pointing
/// into an embedded field of a heap-allocated, variant-specific
/// `ConcreteContext` wrapper. The caller owns that wrapper: comptime-aware
/// callers recover it via `@fieldParentPtr("interface", ctx)` and call
/// the variant's `deinit`; polymorphic borrowers (e.g. `ChatSession`)
/// use the `*Context` and never deinit it.
pub fn createContext(
    self: *Model,
    io: std.Io,
    allocator: std.mem.Allocator,
    options: Context.Options,
) !*Context {
    return try self.vtable.createContext(self, io, allocator, options);
}

/// Tear down the variant's full allocation. After this, the Model (and
/// every pointer borrowed from it — tokenizer, contexts) is dangling.
pub fn deinit(self: *Model) void {
    self.vtable.destroy(self);
}

/// True iff `token` ends a turn in this model. Raw-completion callers
/// stop on the tokenizer's designated EOS; chat-capable models may also
/// emit distinct end-of-turn markers (Qwen's `<|im_end|>`, Llama 3's
/// `<|eot_id|>`) which `Chat.isEndOfTurn` knows about.
pub fn isEndOfTurn(self: *const Model, token: u32) bool {
    if (token == self.tokenizer.eos_token_id) return true;
    if (self.chat) |c| return c.isEndOfTurn(token);
    return false;
}

const std = @import("std");
const Tokenizer = @import("tokenizer.zig");
const Sampler = @import("sampler.zig");
const Context = @import("context.zig");
const Chat = @import("chat.zig");
