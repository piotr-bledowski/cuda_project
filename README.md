# cuda_project

GPU-accelerated **2D lattice Boltzmann (D2Q9)** fluid simulation. Two chambers with different density are separated by a wall with a central hole; flow develops from the pressure/density gradient. The physics run in **CUDA**; **SDL3** handles the window, keyboard input, and drawing density and velocity fields.

**Controls:** Space — open/close the hole · Close window — quit

---

## Requirements (Windows)

| Component | Notes |
|-----------|--------|
| [NVIDIA GPU](https://www.nvidia.com/drivers) + driver | CUDA-capable GPU; driver must support your toolkit version (`nvidia-smi`) |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | Install the version you need; note its [supported MSVC versions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#supported-vs-versions) |
| [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) | Workload: **Desktop development with C++** (MSVC, Windows SDK) |
| [CMake](https://cmake.org/download/) | 3.18+; add to PATH |
| [SDL3](https://github.com/libsdl-org/SDL/releases) | Developer package (see below) |

Optional but recommended: **Ninja** (bundled with VS under `Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja`, or via CMake).

The repository ships **`main.cu`** only. You must add `CMakeLists.txt`, download SDL3, and configure paths yourself (`build/`, `SDL3-*`, and local helper scripts are gitignored).

---

## Setup

### 1. SDL3

1. Open [SDL releases](https://github.com/libsdl-org/SDL/releases).
2. Download **`SDL3-devel-*-VC.zip`** (Visual C++ / x64).
3. Extract into the project root, e.g. `SDL3-3.4.8\` (folder name may vary).
4. CMake will need: `SDL3_DIR=<path-to-extracted>/cmake`

### 2. `CMakeLists.txt`

Create `CMakeLists.txt` in the project root:

```cmake
cmake_minimum_required(VERSION 3.18)
project(lbm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Set to your GPU architecture, e.g. 75 (RTX 20xx), 86 (RTX 30xx), 89 (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(SDL3 REQUIRED)

add_executable(main main.cu)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(main PRIVATE SDL3::SDL3)

if(WIN32)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL3::SDL3>
            $<TARGET_FILE_DIR:main>
    )
endif()
```

Adjust `CMAKE_CUDA_ARCHITECTURES` if `native` is unsupported by your CMake/CUDA pair; use an explicit value from [NVIDIA’s list](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list).

### 3. Environment

Use a **plain `cmd.exe`** (not a conda-heavy shell) so `PATH` stays short.

Set toolkit paths (edit version folder names to match your install):

```bat
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y
set CUDA_PATH_VX_Y=%CUDA_PATH%
set NVCC_PREPEND_FLAGS=--use-local-env
```

If your CUDA version requires an older MSVC than your default VS toolset, use [vcvars](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line) with the version your CUDA docs allow, e.g.:

```bat
call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39.33519
```

(`-vcvars_ver=14.39` alone often fails; use the full folder name under `VC\Tools\MSVC\`.)

### 4. Configure and build

```bat
cd <project-root>
mkdir build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DSDL3_DIR="<project-root>\SDL3-3.4.8\cmake"

cmake --build .
```

**Run** (Ninja puts the exe in `build\`, not `build\Release\`):

```bat
main.exe
```

Keep `SDL3.dll` next to `main.exe` (the post-build step above copies it).

---

## Troubleshooting (Windows)

| Symptom | What to try |
|---------|-------------|
| `CudaToolkitDir ''` / toolkit not found | Set `CUDA_PATH` (and `CUDA_PATH_V*_*` matching your folder name); restart the terminal |
| `unsupported Microsoft Visual Studio version` / `expected CUDA X.Y or newer` | CUDA and MSVC versions mismatch — install a [compatible pair](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#supported-vs-versions) or an older MSVC toolset via VS Installer |
| `The input line is too long` (nvcc / vcvars) | Shorten `PATH` (fresh `cmd.exe`) and set `NVCC_PREPEND_FLAGS=--use-local-env` |
| `Compiler 'cl.exe' in PATH different than -ccbin` | Do not mix dev shells; use one consistent `cl.exe` on `PATH` with `--use-local-env` |
| `LNK1104: cannot open msvcprt*.lib` / `LIBCMT.lib` | Incomplete MSVC install — repair C++ build tools in VS Installer; ensure `LIB` includes `...\MSVC\<ver>\lib\x64` |
| `cmake` / `ninja` not found | Add [CMake](https://cmake.org/download/) and Ninja to `PATH` |
| No `main.exe` after “success” | Run `cmake --build .`; with Ninja look in `build\main.exe`, not `build\Release\` |
| Black window / crash at start | Update GPU driver; confirm `nvidia-smi` CUDA version ≥ toolkit version |

---

## Reference

- [CUDA Installation Guide (Windows)](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [SDL3](https://wiki.libsdl.org/SDL3/Installation)
- [CMake CUDA support](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)
