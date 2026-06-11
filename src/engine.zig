//! Per-variant metadata and `Context` factory, embedded as a field on each
//! variant's concrete model type and reached via `Model.engine`.
//! `createContext` is the entry point; see its doc for the ownership contract.

vocabulary_size: usize,
max_len: usize,
vtable: *const VTable,
/// Per-family preset table keyed by `Sampler.Profile`. Borrowed —
/// typically a `pub const` in the family's `common/sampler_defaults.zig`.
/// Null when the family has not wired presets.
sampler_presets: ?*const @import("sampler.zig").Presets = null,

const Engine = @This();

// In its own file so this VTable doesn't collide with Context.VTable.
pub const VTable = struct {
    /// Heap-allocate a `ConcreteContext`, populate its embedded
    /// `Context` (including the `Context.VTable` pointer), and return
    /// `&concrete.context`. The caller owns the returned pointer; see
    /// `Engine.createContext` for the ownership contract.
    createContext: *const fn (
        *Engine,
        std.Io,
        std.mem.Allocator,
        Tokenizer,
        Context.Options,
    ) anyerror!*Context,

    /// Tear down the variant's full allocation — weights, caches,
    /// thread pool, and family aggregate. Polymorphic callers that
    /// own the Model's lifetime call this on shutdown via
    /// `Engine.destroy`. Variants whose lifetime is
    /// stack- or caller-managed install a no-op so the slot is always
    /// safe to invoke; whether to call it is the caller's choice.
    destroy: *const fn (*Engine) void,

    /// Optional tokenizer override. Variants whose tokenizer state lives
    /// outside the runtime's `Tokenizer` (e.g. wrappers around an
    /// external library that owns its own vocabulary) install these so
    /// `Context.prefill` and other call sites route through the variant's
    /// native tokenizer instead of the embedded `Tokenizer`. When `null`,
    /// `Context.encode` / `Context.decode` fall back to the embedded
    /// `Tokenizer.encode` / `.decode`. Returned slices are owned by the
    /// caller and freed via `allocator`.
    encode: ?*const fn (*Engine, std.mem.Allocator, []const u8) anyerror![]const u32 = null,
    decode: ?*const fn (*Engine, std.mem.Allocator, []const u32) anyerror![]const u8 = null,
};

/// Create a live `Context` for a new conversation. Returns a `*Context`
/// pointing into an embedded field of a heap-allocated, variant-specific
/// `ConcreteContext` wrapper. The caller owns that wrapper: comptime-aware
/// callers recover it via `@fieldParentPtr("context", ctx)` and call the
/// variant's `deinit`; polymorphic borrowers (e.g. `ChatSession`) use the
/// `*Context` and never deinit it.
pub fn createContext(
    self: *Engine,
    io: std.Io,
    allocator: std.mem.Allocator,
    tokenizer: Tokenizer,
    options: Context.Options,
) !*Context {
    return try self.vtable.createContext(self, io, allocator, tokenizer, options);
}

/// Tear down the variant's full allocation. Calling this is the
/// caller's choice — variants that don't allocate (stack-allocated,
/// caller-managed) install a no-op.
pub fn destroy(self: *Engine) void {
    self.vtable.destroy(self);
}

/// Encode `text` to tokens using the variant's native tokenizer if it
/// installed an `encode` hook; otherwise fall back to `fallback.encode`.
/// Used by callers that have an `Engine` + a fallback `Tokenizer` but no
/// `Context` yet (e.g. bench seeding a synthetic prefill prompt).
pub fn encode(
    self: *Engine,
    allocator: std.mem.Allocator,
    text: []const u8,
    fallback: Tokenizer,
) ![]const u32 {
    if (self.vtable.encode) |hook| return try hook(self, allocator, text);
    return try fallback.encode(allocator, text);
}

/// Decode `tokens` to text. Mirrors `encode` — vtable hook first, then
/// the fallback `Tokenizer.decode`.
pub fn decode(
    self: *Engine,
    allocator: std.mem.Allocator,
    tokens: []const u32,
    fallback: Tokenizer,
) ![]const u8 {
    if (self.vtable.decode) |hook| return try hook(self, allocator, tokens);
    return try fallback.decode(allocator, tokens);
}

const std = @import("std");
const Context = @import("context.zig");
const Tokenizer = @import("tokenizer.zig");
