vtable: *const VTable,

pub const VTable = struct {
    load: *const fn (std.Io, std.mem.Allocator, []const u8) anyerror!Loaded,
    unload: *const fn (Loaded) void,
    encode: *const fn (Loaded, []const u8) anyerror![]const u32,
    decode: *const fn (Loaded, []const u32) anyerror![]const u8,
    createContext: *const fn (Loaded) anyerror!Context,
    destroyContext: *const fn (Loaded, Context) void,
    prefill: *const fn (Loaded, Context, []const u32) anyerror!void,
    endTurn: *const fn (Loaded, Context) anyerror!void,
    generate: *const fn (Loaded, Context) anyerror!void,
    sample: *const fn (Loaded, Context, ?Sampler.Options) anyerror!u32,
    classifyToken: *const fn (Loaded, u32) Message.Marker,
    formatMessages: *const fn (Loaded, []const Message) anyerror![]const u8,
    parseToolCall: *const fn (Loaded, []const u8) anyerror![]const Message.ToolCall,
};

pub fn load(self: @This(), io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Loaded {
    return self.vtable.load(io, allocator, path);
}

pub fn unload(self: @This(), model: Loaded) void {
    self.vtable.unload(model);
}

pub fn encode(self: @This(), model: Loaded, text: []const u8) ![]const u32 {
    return self.vtable.encode(model, text);
}

pub fn decode(self: @This(), model: Loaded, tokens: []const u32) ![]const u8 {
    return self.vtable.decode(model, tokens);
}

pub fn createContext(self: @This(), model: Loaded) !Context {
    return self.vtable.createContext(model);
}

pub fn destroyContext(self: @This(), model: Loaded, context: Context) void {
    return self.vtable.destroyContext(model, context);
}

pub fn prefill(self: @This(), model: Loaded, context: Context, tokens: []const u32) !void {
    return self.vtable.prefill(model, context, tokens);
}

pub fn endTurn(self: @This(), model: Loaded, context: Context) !void {
    return self.vtable.endTurn(model, context);
}

pub fn generate(self: @This(), model: Loaded, context: Context) !void {
    return self.vtable.generate(model, context);
}

pub fn sample(self: @This(), model: Loaded, context: Context, options: ?Sampler.Options) !u32 {
    return self.vtable.sample(model, context, options);
}

pub fn classifyToken(self: @This(), model: Loaded, token: u32) Message.Marker {
    return self.vtable.classifyToken(model, token);
}

pub fn formatMessages(self: @This(), model: Loaded, messages: []const Message) ![]const u8 {
    return self.vtable.formatMessages(model, messages);
}

pub fn parseToolCall(self: @This(), model: Loaded, message: []const u8) ![]const Message.ToolCall {
    return self.vtable.parseToolCall(model, message);
}

const Model = @This();
pub const Loaded = *anyopaque;
pub const Context = *anyopaque;

const std = @import("std");

pub const Message = @import("message.zig").Message;
pub const Sampler = @import("sampler.zig");
