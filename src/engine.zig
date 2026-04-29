//! Per-variant metadata + Context factory. Embedded as a field on each
//! variant's concrete model type and pointed at by `Model.engine`.
//!
//! Callers obtain a `Context` by calling `engine.createContext(...)`.
//! The factory returns a `*Context` — a pointer into an embedded
//! `Context` field inside a variant-specific `ConcreteContext` wrapper
//! the factory heap-allocated. The caller owns the wrapper; when
//! comptime-aware code (e.g. the harness) knows the variant type, it can
//! `@fieldParentPtr("context", ctx)` to recover `*ConcreteContext` and
//! call its variant-specific `deinit`. Polymorphic borrowers (e.g.
//! `ChatSession`) just use `*Context` and do not deinit it.
//!
//! Separate file rather than nested in `context.zig` because its own
//! `VTable` struct would otherwise collide with `Context.VTable`. Still a
//! vtable pattern for consistency with `Context` — even with a single
//! method today, grouping behind `vtable: *const VTable` keeps the shape
//! uniform and leaves room to grow.

vocabulary_size: usize,
max_len: usize,
vtable: *const VTable,
/// Optional teardown: tear down the variant's full allocation —
/// weights, caches, thread pool, family aggregate, etc. Polymorphic
/// callers (e.g. harness-v2 runners) that own the Model's lifetime
/// call this on shutdown. Variants that are stack-allocated or whose
/// lifetime is caller-managed can leave it null; the harness treats
/// missing `destroy` as "caller handles it" and skips the call.
destroy: ?*const fn (*Engine) void = null,

const Engine = @This();

pub const VTable = struct {
    /// Heap-allocate a `ConcreteContext`, populate its embedded
    /// `Context` (including the `Context.VTable` pointer), and return
    /// `&concrete.context`. The caller owns the returned pointer; see
    /// the file docstring for the ownership contract.
    createContext: *const fn (
        *Engine,
        std.Io,
        std.mem.Allocator,
        Tokenizer,
        Context.Options,
    ) anyerror!*Context,
};

pub fn createContext(
    self: *Engine,
    io: std.Io,
    allocator: std.mem.Allocator,
    tokenizer: Tokenizer,
    options: Context.Options,
) !*Context {
    return try self.vtable.createContext(self, io, allocator, tokenizer, options);
}

const std = @import("std");
const Context = @import("context.zig");
const Tokenizer = @import("tokenizer.zig");
