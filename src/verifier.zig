pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: std.process.Environ,
    Target: type,
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

        var target: Target = try .init(io, allocator, model_path);
        defer target.deinit();
        const model: *Model = &target.interface;

        const t_init_end = std.Io.Clock.now(.awake, io);
        const init_ms: u64 = @intCast(t_init_start.durationTo(t_init_end).toMilliseconds());
        try stdout_writer.print("Load time: {d}ms\n", .{init_ms});

        for (test_suite.test_cases) |test_case| {
            const t_testcase_start = std.Io.Clock.now(.awake, io);

            const context = try model.createContext();
            defer model.destroyContext(context);

            var response: std.ArrayList(u8) = .empty;
            defer response.deinit(allocator);

            const t_prefill_start = std.Io.Clock.now(.awake, io);
            const prefill_len = try model.prefillMessages(context, test_case.input);
            const t_prefill_end = std.Io.Clock.now(.awake, io);
            const prefill_ns: u64 = @intCast(t_prefill_start.durationTo(t_prefill_end).nanoseconds);

            const t_decode_start = std.Io.Clock.now(.awake, io);
            var tokens_decoded: usize = 0;

            var tool_call_buffer: std.ArrayList(u8) = .empty;
            var tool_calls: std.ArrayList(Message.ToolCall) = .empty;
            var is_tool_call: bool = false;
            while (true) {
                const next = try model.next(context);
                defer allocator.free(next.word);
                tokens_decoded += 1;

                switch (next.marker) {
                    .turn_end => break,
                    .tool_call_start => {
                        is_tool_call = true;
                        tool_call_buffer.shrinkAndFree(allocator, 0);
                    },
                    .tool_call_end => {
                        is_tool_call = false;
                        const tool_call = try model.parseToolCall(tool_call_buffer.items);
                        try tool_calls.appendSlice(allocator, tool_call);
                    },
                    .content => {
                        if (is_tool_call) {
                            try tool_call_buffer.appendSlice(allocator, next.word);
                        }
                    },
                    else => {},
                }

                try response.appendSlice(allocator, next.word);
            }
            const t_decode_end = std.Io.Clock.now(.awake, io);
            const decode_ns: u64 = @intCast(t_decode_start.durationTo(t_decode_end).nanoseconds);

            var success: bool = true;

            if (!std.mem.eql(u8, test_case.expected, response.items)) {
                try stdout_writer.print("{s}: FAILED!\n", .{test_case.name});
                try stdout_writer.print("Whole interation:\n", .{});
                try stdout_writer.print("{s}\n", .{try model.formatMessages(test_case.input)});
                try stdout_writer.print("Received response:\n", .{});
                try stdout_writer.flush();

                try stderr_writer.print("{s}", .{response.items});
                try stderr_writer.flush();

                try stdout_writer.print("\n", .{});
                try stdout_writer.flush();
                success = false;
            }

            if (success & !try compareToolCalls(stdout_writer, test_case.expectedToolCalls, tool_calls.items)) {
                try stdout_writer.print("{s}: FAILED!\n", .{test_case.name});
                try stdout_writer.flush();
                success = false;
            }

            if (success) {
                const prefill_s = @as(f64, @floatFromInt(prefill_ns)) / std.time.ns_per_s;
                const decode_s = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s;
                const prefil_tk_s = @as(f64, @floatFromInt(prefill_len)) / prefill_s;
                const decode_tk_s = @as(f64, @floatFromInt(tokens_decoded)) / decode_s;

                const t_testcase_end = std.Io.Clock.now(.awake, io);
                const testcase_ms: u64 = @intCast(t_testcase_start.durationTo(t_testcase_end).toMilliseconds());

                try stdout_writer.print(
                    "{s}: PASSED! pp{d}: {d:.1}tk/s,  tg{d}: {d:.1}tk/s, total: {d}ms\n",
                    .{
                        test_case.name,
                        prefill_len,
                        prefil_tk_s,
                        tokens_decoded,
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
