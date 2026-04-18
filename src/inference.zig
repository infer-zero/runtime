//! The tight inference core: a type-erased handle plus a vtable of the six
//! hot methods every model variant must implement. No chat awareness, no
//! tool awareness, no special-token data — those all live on the aggregate
//! `Model` (which *has* an `Inference`) and on `Chat` / `Tool`.
//!
//! `Runtime` drives an `Inference` directly. `ChatSession` layers on top of
//! `Runtime` and brings in chat/tool concerns via the aggregate.

ptr: *anyopaque,
vtable: *const VTable,

const Inference = @This();

pub const VTable = struct {
    deinit: *const fn (*Inference) void,
    vocabulary: *const fn (*Inference) Vocabulary,
    createContext: *const fn (*Inference) anyerror!*anyopaque,
    destroyContext: *const fn (*Inference, *anyopaque) void,
    prefill: *const fn (*Inference, *anyopaque, []const u32) anyerror!void,
    next: *const fn (*Inference, *anyopaque, u32) anyerror![]const f32,
};

pub fn deinit(self: *Inference) void {
    self.vtable.deinit(self);
}

pub fn vocabulary(self: *Inference) Vocabulary {
    return self.vtable.vocabulary(self);
}

pub fn createContext(self: *Inference) !*anyopaque {
    return try self.vtable.createContext(self);
}

pub fn destroyContext(self: *Inference, ctx: *anyopaque) void {
    self.vtable.destroyContext(self, ctx);
}

pub fn prefill(self: *Inference, ctx: *anyopaque, tokens: []const u32) !void {
    try self.vtable.prefill(self, ctx, tokens);
}

pub fn next(self: *Inference, ctx: *anyopaque, token: u32) ![]const f32 {
    return try self.vtable.next(self, ctx, token);
}

/// Type-erase a concrete implementation `T` into an `Inference`. `T` must
/// expose `deinit`, `vocabulary`, `createContext`, `destroyContext`, `prefill`,
/// `next`, and a `Context` type. Only the six core methods are wrapped here —
/// chat/tool concerns attach separately on the aggregate `Model`.
pub fn wrap(comptime T: type) type {
    return struct {
        ptr: *T,

        pub fn init(ptr: *T) @This() {
            return .{ .ptr = ptr };
        }

        pub fn inference(self: @This()) Inference {
            return .{ .ptr = self.ptr, .vtable = &vtable_instance };
        }

        const vtable_instance: VTable = .{
            .deinit = deinitFn,
            .vocabulary = vocabularyFn,
            .createContext = createContextFn,
            .destroyContext = destroyContextFn,
            .prefill = prefillFn,
            .next = nextFn,
        };

        fn deinitFn(inf: *Inference) void {
            const concrete: *T = @ptrCast(@alignCast(inf.ptr));
            concrete.deinit();
        }

        fn vocabularyFn(inf: *Inference) Vocabulary {
            const concrete: *T = @ptrCast(@alignCast(inf.ptr));
            return concrete.vocabulary();
        }

        fn createContextFn(inf: *Inference) anyerror!*anyopaque {
            const concrete: *T = @ptrCast(@alignCast(inf.ptr));
            const ctx = try concrete.createContext();
            return @ptrCast(ctx);
        }

        fn destroyContextFn(inf: *Inference, ctx: *anyopaque) void {
            const concrete: *T = @ptrCast(@alignCast(inf.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            concrete.destroyContext(typed_ctx);
        }

        fn prefillFn(inf: *Inference, ctx: *anyopaque, tokens: []const u32) anyerror!void {
            const concrete: *T = @ptrCast(@alignCast(inf.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            try concrete.prefill(typed_ctx, tokens);
        }

        fn nextFn(inf: *Inference, ctx: *anyopaque, token: u32) anyerror![]const f32 {
            const concrete: *T = @ptrCast(@alignCast(inf.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            return try concrete.next(typed_ctx, token);
        }
    };
}

const Vocabulary = @import("vocabulary.zig");
