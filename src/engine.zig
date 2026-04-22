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

const Engine = @This();

pub const VTable = struct {
    /// Heap-allocate a `ConcreteContext`, populate its embedded
    /// `Context` (including the `Context.VTable` pointer), and return
    /// `&concrete.context`. The caller owns the returned pointer; see
    /// the file docstring for the ownership contract.
    createContext: *const fn (
        *Engine,
        std.mem.Allocator,
        Tokenizer,
        Context.Options,
    ) anyerror!*Context,
};

pub fn createContext(
    self: *Engine,
    allocator: std.mem.Allocator,
    tokenizer: Tokenizer,
    options: Context.Options,
) !*Context {
    return try self.vtable.createContext(self, allocator, tokenizer, options);
}

const std = @import("std");
const Context = @import("context.zig");
const Tokenizer = @import("tokenizer.zig");
