pub fn resolve(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: std.process.Environ,
    vendor: []const u8,
    repo: []const u8,
    file: []const u8,
    auth_token: ?[]const u8,
) ![]const u8 {
    const cache_path = try getCachePath(allocator, environ, "infer");
    defer allocator.free(cache_path);

    const target = try std.fs.path.join(allocator, &.{ cache_path, vendor, repo, file });
    errdefer allocator.free(target);

    _ = std.Io.Dir.cwd().statFile(io, target, .{}) catch |stat_err| {
        switch (stat_err) {
            error.FileNotFound => {
                const url = try std.fmt.allocPrint(
                    allocator,
                    "https://huggingface.co/{s}/{s}/resolve/main/{s}",
                    .{
                        vendor,
                        repo,
                        file,
                    },
                );
                defer allocator.free(url);

                download(io, allocator, url, target, auth_token) catch |download_err| {
                    return download_err;
                };

                return target;
            },
            else => return stat_err,
        }
    };
    return target;
}

fn download(
    io: std.Io,
    allocator: std.mem.Allocator,
    url: []const u8,
    dest_path: []const u8,
    auth_token: ?[]const u8,
) !void {
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    const uri = try std.Uri.parse(url);
    const auth_headers = try priv_headers(allocator, auth_token);

    var req = try client.request(
        .GET,
        uri,
        .{
            .redirect_behavior = @enumFromInt(3),
            .privileged_headers = auth_headers,
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
        },
    );
    defer req.deinit();

    try req.sendBodiless();

    const read_buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(read_buffer);

    var response = try req.receiveHead(read_buffer);
    if (response.head.status != .ok) {
        return error.DownloadFailed;
    }

    const reader = response.reader(read_buffer);

    const write_buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(write_buffer);

    var file = try std.Io.Dir.cwd().createFile(io, dest_path, .{});
    defer file.close(io);
    var file_writer = file.writer(io, write_buffer);
    const writer = &file_writer.interface;

    _ = try reader.streamRemaining(writer);
    try file_writer.interface.flush();
}

fn priv_headers(allocator: std.mem.Allocator, auth_token: ?[]const u8) ![]const std.http.Header {
    if (auth_token) |auth| {
        const headers = try allocator.alloc(std.http.Header, 1);
        errdefer allocator.free(headers);

        const auth_value = try std.fmt.allocPrint(allocator, "Bearer: {s}", .{auth});
        errdefer allocator.free(auth_value);

        const auth_name = try allocator.dupe(u8, "Authentication");
        errdefer allocator.free(auth_name);

        headers[0] = .{
            .name = auth_name,
            .value = auth_value,
        };

        return headers;
    }
    return &.{};
}

fn free_headers(allocator: std.mem.Allocator, headers: []const std.http.Header) void {
    if (headers.len > 0) {
        for (headers) |header| {
            allocator.free(header.name);
            allocator.free(header.value);
        }
        allocator.free(headers);
    }
}

fn getCachePath(
    allocator: std.mem.Allocator,
    environ: std.process.Environ,
    program_name: []const u8,
) ![]const u8 {
    switch (builtin.os.tag) {
        .macos => {
            const home = environ.getPosix("HOME") orelse return error.NoCacheDir;
            return std.fs.path.join(allocator, &.{ home, "/Library/Caches/", program_name, "models" });
        },
        .windows => {
            const local = environ.getPosix.get("LOCALAPPDATA") orelse return error.NoCacheDir;
            return std.fs.path.join(allocator, &.{ local, program_name, "models" });
        },
        else => {
            if (environ.getPosix("XDG_CACHE_HOME")) |xdg| {
                return std.fs.path.join(allocator, &.{ xdg, program_name, "models" });
            }
            const home = environ.getPosix("HOME") orelse return error.NoCacheDir;
            return std.fs.path.join(allocator, &.{ home, ".cache", program_name, "models" });
        },
    }
}

const std = @import("std");
const builtin = @import("builtin");
