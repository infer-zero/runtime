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
            while (true) {
                try model.generate(context);

                const next_token = try model.sample(context, greedy);
                tokens_decoded += 1;
                if (model.classify(next_token) == .turn_end) break;

                const next_word = try model.decode(&.{next_token});
                defer allocator.free(next_word);
                try response.appendSlice(allocator, next_word);
            }
            const t_decode_end = std.Io.Clock.now(.awake, io);
            const decode_ns: u64 = @intCast(t_decode_start.durationTo(t_decode_end).nanoseconds);

            if (std.mem.eql(u8, test_case.expected, response.items)) {
                try stdout_writer.print("{s}: PASSED!\n", .{test_case.name});
            } else {
                try stdout_writer.print("{s}: FAILED!\n", .{test_case.name});
                try stdout_writer.flush();

                try stderr_writer.print("{s}", .{response.items});
                try stderr_writer.flush();

                try stdout_writer.print("\n", .{});
                try stdout_writer.flush();
                return error.TestFailed;
            }

            const prefill_s = @as(f64, @floatFromInt(prefill_ns)) / std.time.ns_per_s;
            const decode_s = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s;
            const prefil_tk_s = @as(f64, @floatFromInt(prefill_len)) / prefill_s;
            const decode_tk_s = @as(f64, @floatFromInt(tokens_decoded)) / decode_s;

            try stdout_writer.print("- pp{d}: {d:.1}tk/s\n", .{ prefill_len, prefil_tk_s });
            try stdout_writer.print("- tg{d}: {d:.1}tk/s\n", .{ tokens_decoded, decode_tk_s });

            const t_testcase_end = std.Io.Clock.now(.awake, io);
            const testcase_ms: u64 = @intCast(t_testcase_start.durationTo(t_testcase_end).toMilliseconds());
            try stdout_writer.print("- Total: {d}ms\n", .{testcase_ms});

            try stdout_writer.flush();
        }
    }
}

pub const TestSuite = struct {
    model: struct {
        vendor: []const u8,
        repository: []const u8,
        file: []const u8,
    },
    test_cases: []const TestCase,
};

pub const TestCase = struct {
    name: []const u8,
    input: []const Message,
    expected: []const u8,
};

const greedy: Sampler.Options = .{
    .temperature = 0.0,
    .top_k = 50,
    .top_p = 0.1,
    .min_p = 0.0,
    .repetition_penalty = 1.05,
    .repetition_penalty_last_n = 64,
};

const std = @import("std");
const Model = @import("model.zig");
const Sampler = @import("sampler.zig");
const Message = @import("message.zig").Message;
const download = @import("download.zig");
