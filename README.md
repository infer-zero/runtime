# infer-runtime

This is an interface and cruntime for implementing inference models in Zig.

Provides the foundation as well as an interface with the minimal necessary for a model to offer inference.

Some AI assistance was used in writing this code.

## Usage

Fetch the library:

```bash
zig fetch --save git+https://github.com/infer-zero/runtime
```

Add the model in your `build.zig`:

```zig
const runtime_dep = b.dependency("infer_runtime", .{ .target = target, .optimize = optimize });
my_mod.addImport("runtime", runtime_dep.module("infer_runtime"));
```

### Running inference

Code is still changing, so I will update here once I have a more stable interface.

For now take a look at [src/verifier.zig] or some of the models implementation.

## License

MIT
