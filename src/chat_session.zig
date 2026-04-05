allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
runtime: *Runtime,
context: Runtime.Context,
messages: std.ArrayListUnmanaged(Message),
options: Model.ChatOptions,
state: State = .content,
content_buf: std.ArrayListUnmanaged(u8) = .empty,
thinking_buf: std.ArrayListUnmanaged(u8) = .empty,
tool_call_buf: std.ArrayListUnmanaged(u8) = .empty,
pending_end_of_turn: bool = false,

const State = enum { content, thinking, tool_call };

pub const Event = union(enum) {
    content: []const u8,
    thinking: []const u8,
    tool_call: []const u8,
    thinking_start,
    thinking_end,
    tool_call_start,
    tool_call_end,
    end_of_turn,
};

pub fn init(allocator: std.mem.Allocator, runtime: *Runtime, options: Model.ChatOptions) !@This() {
    const context = try runtime.start();
    return .{
        .allocator = allocator,
        .arena = std.heap.ArenaAllocator.init(allocator),
        .runtime = runtime,
        .context = context,
        .messages = .empty,
        .options = options,
    };
}

pub fn deinit(self: *@This()) void {
    self.context.deinit();
    self.arena.deinit();
}

pub fn addSystemMessage(self: *@This(), text: []const u8) !void {
    const arena_alloc = self.arena.allocator();
    const owned = try arena_alloc.dupe(u8, text);
    try self.messages.append(arena_alloc, .{ .system = owned });
}

pub fn send(self: *@This(), user_text: []const u8) !void {
    const arena_alloc = self.arena.allocator();
    const owned = try arena_alloc.dupe(u8, user_text);
    try self.messages.append(arena_alloc, .{ .user = owned });

    const prompt = try self.runtime.model.chatFormat(
        self.allocator,
        self.messages.items,
        self.options,
    );
    defer self.allocator.free(prompt);

    // Reset context for full re-prefill.
    self.context.deinit();
    self.context = try self.runtime.start();

    try self.context.prefill(prompt);

    // Reset generation state.
    self.state = .content;
    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
    self.pending_end_of_turn = false;
}

pub fn next(self: *@This()) !?Event {
    // Emit deferred end_of_turn after finalizing the turn.
    if (self.pending_end_of_turn) {
        self.pending_end_of_turn = false;
        return null;
    }

    const text = self.context.next() catch |err| {
        if (err == error.ContextFull) {
            try self.finalizeTurn();
            return .end_of_turn;
        }
        return err;
    } orelse {
        // EOS — finalize turn.
        try self.finalizeTurn();
        return .end_of_turn;
    };

    const token_class = self.runtime.model.classifyToken(self.context.current_token);

    switch (token_class) {
        .thinking_start => {
            self.state = .thinking;
            self.allocator.free(text);
            return .thinking_start;
        },
        .thinking_end => {
            self.state = .content;
            self.allocator.free(text);
            return .thinking_end;
        },
        .tool_call_start => {
            self.state = .tool_call;
            self.allocator.free(text);
            return .tool_call_start;
        },
        .tool_call_end => {
            self.state = .content;
            self.allocator.free(text);
            return .tool_call_end;
        },
        .end_of_turn => {
            self.allocator.free(text);
            try self.finalizeTurn();
            self.pending_end_of_turn = true;
            return .end_of_turn;
        },
        .content => {
            const arena_alloc = self.arena.allocator();
            switch (self.state) {
                .thinking => {
                    try self.thinking_buf.appendSlice(arena_alloc, text);
                    return .{ .thinking = text };
                },
                .content => {
                    try self.content_buf.appendSlice(arena_alloc, text);
                    return .{ .content = text };
                },
                .tool_call => {
                    try self.tool_call_buf.appendSlice(arena_alloc, text);
                    return .{ .tool_call = text };
                },
            }
        },
    }
}

pub fn supplyToolResult(self: *@This(), tool_call_id: []const u8, content: []const u8) !void {
    const arena_alloc = self.arena.allocator();

    // Finalize the current assistant turn if not already done.
    if (self.content_buf.items.len > 0 or self.thinking_buf.items.len > 0 or self.tool_call_buf.items.len > 0) {
        try self.finalizeTurn();
    }

    // Append tool result.
    try self.messages.append(arena_alloc, .{ .tool_result = .{
        .tool_call_id = try arena_alloc.dupe(u8, tool_call_id),
        .content = try arena_alloc.dupe(u8, content),
    } });

    // Re-prefill with full history.
    const prompt = try self.runtime.model.chatFormat(
        self.allocator,
        self.messages.items,
        self.options,
    );
    defer self.allocator.free(prompt);

    self.context.deinit();
    self.context = try self.runtime.start();
    try self.context.prefill(prompt);

    self.state = .content;
    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
    self.pending_end_of_turn = false;
}

fn finalizeTurn(self: *@This()) !void {
    const arena_alloc = self.arena.allocator();

    const thinking: ?[]const u8 = if (self.thinking_buf.items.len > 0)
        try arena_alloc.dupe(u8, self.thinking_buf.items)
    else
        null;

    const content = if (self.content_buf.items.len > 0)
        try arena_alloc.dupe(u8, self.content_buf.items)
    else
        "";

    try self.messages.append(arena_alloc, .{ .assistant = .{
        .content = content,
        .thinking = thinking,
    } });

    self.content_buf = .empty;
    self.thinking_buf = .empty;
    self.tool_call_buf = .empty;
}

const std = @import("std");
const Runtime = @import("runtime.zig");
const Model = @import("model.zig");
const Message = @import("message.zig").Message;
