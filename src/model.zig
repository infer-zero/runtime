allocator: std.mem.Allocator,
vtable: *const VTable,

pub const VTable = struct {
    encode: *const fn (*Model, []const u8) anyerror![]const u32,
    decode: *const fn (*Model, []const u32) anyerror![]const u8,
    createContext: *const fn (*Model) anyerror!Context,
    destroyContext: *const fn (*Model, Context) void,
    prefill: *const fn (*Model, Context, []const u32) anyerror!void,
    generate: *const fn (*Model, Context) anyerror!void,
    sample: *const fn (*Model, Context, ?Sampler.Options) anyerror!u32,
    classify: *const fn (*Model, u32) Marker,
    format: *const fn (*Model, []const Message) anyerror![]const u8,
};

pub fn encode(self: *@This(), text: []const u8) ![]const u32 {
    return self.vtable.encode(self, text);
}

pub fn decode(self: *@This(), tokens: []const u32) ![]const u8 {
    return self.vtable.decode(self, tokens);
}

pub fn createContext(self: *@This()) !Context {
    return self.vtable.createContext(self);
}

pub fn destroyContext(self: *@This(), context: Context) void {
    return self.vtable.destroyContext(self, context);
}

pub fn prefill(self: *@This(), context: Context, tokens: []const u32) !void {
    return self.vtable.prefill(self, context, tokens);
}

/// This will format with chat template, encode with tokenizer and prefill the tokens.
pub fn prefillMessages(
    self: *@This(),
    context: Context,
    messages: []const Message,
) !usize {
    const formatted = try self.format(messages);
    defer self.allocator.free(formatted);

    const tokens = try self.encode(formatted);
    defer self.allocator.free(tokens);

    try self.prefill(context, tokens);

    return tokens.len;
}

pub fn generate(self: *@This(), context: Context) !void {
    return self.vtable.generate(self, context);
}

pub fn sample(self: *@This(), context: Context, options: ?Sampler.Options) !u32 {
    return self.vtable.sample(self, context, options);
}

pub fn classify(self: *@This(), token: u32) Marker {
    return self.vtable.classify(self, token);
}

pub fn format(self: *@This(), messages: []const Message) ![]const u8 {
    return self.vtable.format(self, messages);
}

pub fn next(
    self: *@This(),
    context: Context,
) !struct {
    marker: Marker,
    word: []const u8,
} {
    try self.generate(context);
    const token = try self.sample(context, null);
    const word = try self.decode(&.{token});
    const marker = self.classify(token);
    return .{ .marker = marker, .word = word };
}

const Model = @This();
pub const Context = *anyopaque;

const std = @import("std");

pub const Message = @import("message.zig").Message;
pub const Marker = @import("message.zig").Marker;
pub const Sampler = @import("sampler.zig");
