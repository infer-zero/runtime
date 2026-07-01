pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: std.process.Environ,
    model: Model,
    test_suites: []const TestSuite,
) !void {
    const stdout_file = std.Io.File.stdout();
    const stdout_buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(stdout_buffer);
    var stdout_file_writer = stdout_file.writer(io, stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    const stderr_file = std.Io.File.stderr();
    const stderr_buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(stderr_buffer);
    var stderr_file_writer = stderr_file.writer(io, stderr_buffer);
    const stderr_writer = &stderr_file_writer.interface;

    for (test_suites) |test_suite| {
        try stdout_writer.print(
            "Finding model {s}/{s}/{s}...\n",
            .{
                test_suite.model.vendor,
                test_suite.model.repository,
                test_suite.model.file,
            },
        );
        try stdout_writer.flush();

        const model_path = try download.resolve(
            io,
            allocator,
            environ,
            test_suite.model.vendor,
            test_suite.model.repository,
            test_suite.model.file,
            environ.getPosix("HF_TOKEN"),
        );
        defer allocator.free(model_path);
        try stdout_writer.print("Using model: {s}\n", .{model_path});

        const t_init_start = std.Io.Clock.now(.awake, io);

        const loaded_model = try model.load(io, allocator, model_path);
        defer model.unload(loaded_model);

        const t_init_end = std.Io.Clock.now(.awake, io);
        const init_ms: u64 = @intCast(t_init_start.durationTo(t_init_end).toMilliseconds());
        try stdout_writer.print("Load time: {d}ms\n", .{init_ms});
        try stdout_writer.flush();

        for (test_suite.test_cases) |test_case| {
            const t_testcase_start = std.Io.Clock.now(.awake, io);

            const context = try model.createContext(loaded_model);
            defer model.destroyContext(loaded_model, context);

            var success: bool = true;

            var prefill_len: usize = 0;
            var prefill_ns: u64 = 0;
            var decode_len: usize = 0;
            var decode_ns: u64 = 0;

            for (test_case.turns) |test_turn| {
                var response: std.ArrayList(u8) = .empty;
                defer response.deinit(allocator);

                const input_formatted = try model.formatMessages(loaded_model, test_turn.input);
                defer allocator.free(input_formatted);

                const input_tokens = try model.encode(loaded_model, input_formatted);
                defer allocator.free(input_tokens);

                const t_prefill_start = std.Io.Clock.now(.awake, io);
                try model.prefill(loaded_model, context, input_tokens);
                const t_prefill_end = std.Io.Clock.now(.awake, io);
                prefill_ns += @intCast(t_prefill_start.durationTo(t_prefill_end).nanoseconds);
                prefill_len += input_tokens.len;

                const t_decode_start = std.Io.Clock.now(.awake, io);

                var tool_call_buffer: std.ArrayList(u8) = .empty;
                var tool_calls: std.ArrayList(Message.ToolCall) = .empty;
                var is_tool_call: bool = false;
                while (true) {
                    try model.generate(loaded_model, context);
                    const token = try model.sample(loaded_model, context, greedy);

                    const word = try model.decode(loaded_model, &.{token});
                    defer allocator.free(word);

                    const marker = model.classifyToken(loaded_model, token);
                    decode_len += 1;

                    switch (marker) {
                        .turn_end => break,
                        .tool_call_start => {
                            is_tool_call = true;
                            tool_call_buffer.shrinkAndFree(allocator, 0);
                        },
                        .tool_call_end => {
                            is_tool_call = false;
                            const tool_call = try model.parseToolCall(loaded_model, tool_call_buffer.items);
                            try tool_calls.appendSlice(allocator, tool_call);
                        },
                        .content => {
                            if (is_tool_call) {
                                try tool_call_buffer.appendSlice(allocator, word);
                            }
                        },
                        else => {},
                    }

                    try response.appendSlice(allocator, word);
                }
                const t_decode_end = std.Io.Clock.now(.awake, io);
                decode_ns += @intCast(t_decode_start.durationTo(t_decode_end).nanoseconds);

                try model.endTurn(loaded_model, context);

                if (!std.mem.eql(u8, test_turn.expected, response.items)) {
                    try stdout_writer.print("{s}: FAILED!\n", .{test_case.name});
                    try stdout_writer.print("Whole interation:\n", .{});
                    try stdout_writer.print("{s}\n", .{input_formatted});
                    try stdout_writer.print("Received response:\n", .{});
                    try stdout_writer.flush();

                    try stderr_writer.print("{s}", .{response.items});
                    try stderr_writer.flush();

                    try stdout_writer.print("\n", .{});
                    try stdout_writer.flush();
                    success = false;
                }

                if (success & !try compareToolCalls(stdout_writer, test_turn.expectedToolCalls, tool_calls.items)) {
                    try stdout_writer.print("{s}: FAILED!\n", .{test_case.name});
                    try stdout_writer.flush();
                    success = false;
                }
            }

            if (success) {
                const prefill_s = @as(f64, @floatFromInt(prefill_ns)) / std.time.ns_per_s;
                const decode_s = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s;
                const prefil_tk_s = @as(f64, @floatFromInt(prefill_len)) / prefill_s;
                const decode_tk_s = @as(f64, @floatFromInt(decode_len)) / decode_s;

                const t_testcase_end = std.Io.Clock.now(.awake, io);
                const testcase_ms: u64 = @intCast(t_testcase_start.durationTo(t_testcase_end).toMilliseconds());

                try stdout_writer.print(
                    "{s}: PASSED! pp{d}: {d:.1}tk/s,  tg{d}: {d:.1}tk/s, total: {d}ms\n",
                    .{
                        test_case.name,
                        prefill_len,
                        prefil_tk_s,
                        decode_len,
                        decode_tk_s,
                        testcase_ms,
                    },
                );
                try stdout_writer.flush();
            } else {
                break;
            }
        }
    }
}

pub const TestSuite = struct {
    model: TestModel,
    test_cases: []const TestCase,
};

pub const TestModel = struct {
    vendor: []const u8,
    repository: []const u8,
    file: []const u8,
};

pub const TestCase = struct {
    name: []const u8,
    turns: []const TestTurn,
};

pub const TestTurn = struct {
    input: []const Message,
    expected: []const u8,
    expectedToolCalls: []const Message.ToolCall = &.{},
};

const greedy: Sampler.Options = .{
    .temperature = 0.0,
    .top_k = 50,
    .top_p = 0.1,
    .min_p = 0.0,
    .repetition_penalty = 1.05,
    .repetition_penalty_last_n = 64,
};

fn compareToolCalls(
    writer: *std.Io.Writer,
    expected_list: []const Message.ToolCall,
    received_list: []const Message.ToolCall,
) !bool {
    if (expected_list.len != received_list.len) {
        try writer.print("Tool call failed: expected {d} found {d}.", .{ expected_list.len, received_list.len });
        return false;
    }

    for (expected_list, 0..) |expected, idx| {
        const received = received_list[idx];
        if (!std.mem.eql(u8, expected.name, received.name)) {
            try writer.print("Exected [{d}].name = `{s}`, received `{s}`.\n", .{ idx, expected.name, received.name });
            return false;
        }
        if (expected.arguments.len != received.arguments.len) {
            try writer.print("Exected [{d}].arguments.len = `{d}`, received `{d}`.\n", .{ idx, expected.arguments.len, received.arguments.len });
        }
        for (expected.arguments, 0..) |expected_arg, arg_idx| {
            const received_arg = received.arguments[arg_idx];
            if (!std.mem.eql(u8, expected_arg.name, received_arg.name)) {
                try writer.print("Exected [{d}].arguments[{d}].name = `{s}`, received `{s}`.\n", .{ idx, arg_idx, expected_arg.name, received_arg.name });
                return false;
            }
            // TODO: test read values
        }
    }

    return true;
}

const std = @import("std");
const Model = @import("model.zig");
const Sampler = @import("sampler.zig");
const Message = @import("message.zig").Message;
const download = @import("download.zig");

const log = std.log.scoped(.infer);
