ptr: *anyopaque,
vtable: *const VTable,
eos_token_id: u32,
vocabulary_size: usize,
max_len: usize,

const Model = @This();

pub const VTable = struct {
    deinit: *const fn (*Model) void,
    vocabulary: *const fn (*Model) Vocabulary,
    createContext: *const fn (*Model) anyerror!*anyopaque,
    destroyContext: *const fn (*Model, *anyopaque) void,
    prefill: *const fn (*Model, *anyopaque, []const u32) anyerror!void,
    next: *const fn (*Model, *anyopaque, u32) anyerror![]const f32,
    chatFormat: *const fn (*Model, Allocator, []const Message) anyerror![]const u8,
};

/// Return a typed wrapper for the given concrete model type.
pub fn wrap(comptime T: type) type {
    return struct {
        ptr: *T,

        pub fn init(ptr: *T) @This() {
            return .{ .ptr = ptr };
        }

        /// Produce the type-erased Model interface.
        pub fn model(self: @This()) Model {
            return .{
                .ptr = self.ptr,
                .vtable = &vtable_instance,
                .eos_token_id = self.ptr.eos_token_id,
                .vocabulary_size = self.ptr.config.vocabulary_size,
                .max_len = self.ptr.config.max_len,
            };
        }

        const vtable_instance: VTable = .{
            .deinit = deinitFn,
            .vocabulary = vocabularyFn,
            .createContext = createContextFn,
            .destroyContext = destroyContextFn,
            .prefill = prefillFn,
            .next = nextFn,
            .chatFormat = chatFormatFn,
        };

        fn deinitFn(m: *Model) void {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            concrete.deinit();
        }

        fn vocabularyFn(m: *Model) Vocabulary {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            return concrete.vocabulary();
        }

        fn createContextFn(m: *Model) anyerror!*anyopaque {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const ctx = try concrete.createContext();
            return @ptrCast(ctx);
        }

        fn destroyContextFn(m: *Model, ctx: *anyopaque) void {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            concrete.destroyContext(typed_ctx);
        }

        fn prefillFn(m: *Model, ctx: *anyopaque, tokens: []const u32) anyerror!void {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            try concrete.prefill(typed_ctx, tokens);
        }

        fn nextFn(m: *Model, ctx: *anyopaque, token: u32) anyerror![]const f32 {
            const concrete: *T = @ptrCast(@alignCast(m.ptr));
            const typed_ctx: *T.Context = @ptrCast(@alignCast(ctx));
            return try concrete.next(typed_ctx, token);
        }

        fn chatFormatFn(_: *Model, allocator: Allocator, messages: []const Message) anyerror![]const u8 {
            return try T.chatFormat(allocator, messages);
        }
    };
}

/// Initialize a concrete model and wrap it in the Model interface.
pub fn init(comptime T: type, allocator: Allocator, path: []const u8) !@This() {
    const concrete = try T.init(allocator, path);
    return wrap(T).init(concrete).model();
}

pub fn deinit(self: *@This()) void {
    self.vtable.deinit(self);
}

pub fn vocabulary(self: *@This()) Vocabulary {
    return self.vtable.vocabulary(self);
}

pub fn createContext(self: *@This()) !*anyopaque {
    return try self.vtable.createContext(self);
}

pub fn destroyContext(self: *@This(), ctx: *anyopaque) void {
    self.vtable.destroyContext(self, ctx);
}

pub fn prefill(self: *@This(), ctx: *anyopaque, tokens: []const u32) !void {
    try self.vtable.prefill(self, ctx, tokens);
}

pub fn next(self: *@This(), ctx: *anyopaque, token: u32) ![]const f32 {
    return try self.vtable.next(self, ctx, token);
}

pub fn chatFormat(self: *@This(), allocator: Allocator, messages: []const Message) ![]const u8 {
    return try self.vtable.chatFormat(self, allocator, messages);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const Vocabulary = @import("vocabulary.zig");
const Message = @import("message.zig");

test "model wrap and runtime integration" {
    const testing = std.testing;
    const Runtime = @import("runtime.zig");

    const MockModel = struct {
        eos_token_id: u32 = 2,
        config: struct { vocabulary_size: usize = 4, max_len: usize = 16 } = .{},
        logits: [4]f32 = .{ 0.0, 0.0, 1.0, 0.0 }, // always predict token 2 (eos)

        pub const Context = struct { dummy: u8 = 0 };

        pub fn deinit(_: *@This()) void {}

        pub fn vocabulary(_: *@This()) Vocabulary {
            return .{
                .merge_index = .{},
                .encoding = .{},
                .decoding = .{},
            };
        }

        pub fn createContext(_: *@This()) !*Context {
            return try testing.allocator.create(Context);
        }

        pub fn destroyContext(_: *@This(), ctx: *Context) void {
            testing.allocator.destroy(ctx);
        }

        pub fn prefill(_: *@This(), _: *Context, _: []const u32) !void {}

        pub fn next(self: *@This(), _: *Context, _: u32) ![]const f32 {
            return &self.logits;
        }

        pub fn chatFormat(_: Allocator, _: []const Message) ![]const u8 {
            return "mock";
        }
    };

    var mock = MockModel{};
    var m = wrap(MockModel).init(&mock).model();

    // Exercise the vtable interface
    var runtime = Runtime.init(testing.allocator, m, .{});
    var ctx = try runtime.start();
    defer ctx.deinit();

    // Prefill with dummy tokens
    try m.prefill(ctx.context, &.{ 0, 1 });

    // Next should return logits via vtable
    const logits = try m.next(ctx.context, 1);
    try testing.expectEqual(@as(usize, 4), logits.len);
    try testing.expectEqual(@as(f32, 1.0), logits[2]);

    // Cleanup
    m.deinit();
}
