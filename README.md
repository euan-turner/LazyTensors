## CMake Options
Default (Release, no debug):

All from build

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