## CMake Options
Default (Release, no debug):

All from build

````markdown
## CMake Options

Default (Release, no debug):

```bash
cmake ..
cmake --build .
```

With host debug:

```bash
cmake .. -DHOST_DEBUG=ON
cmake --build .
```

With host + CUDA debug:

```bash
cmake .. -DHOST_DEBUG=ON -DCUDA_DEBUG=ON
cmake --build .
```

High-level debug switch

You can enable debug symbols using a single option which sets the build type to
Debug and enables host debug flags:

```bash
cmake .. -DENABLE_DEBUG_SYMBOLS=ON
cmake --build .
```

Notes:
- `ENABLE_DEBUG_SYMBOLS` sets `HOST_DEBUG=ON` and `CMAKE_BUILD_TYPE=Debug` unless you
	explicitly set those variables yourself on the command line.
- Use `-DCUDA_DEBUG=ON` if you need device-level CUDA debug info. Be aware that
	`-G` greatly slows CUDA kernels and may change code generation.
- To build Release with host debug flags only, use `-DHOST_DEBUG=ON -DCMAKE_BUILD_TYPE=Release`.
````