cmake_minimum_required(VERSION 3.18)

# ── CUDA 11.8 ─────────────────────────────────────────────────────────────────
set(_CUDA118 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
if(EXISTS "${_CUDA118}/bin/nvcc.exe")
    set(ENV{CUDA_PATH} "${_CUDA118}")
    set(ENV{CUDA_PATH_V11_8} "${_CUDA118}")
    set(CMAKE_CUDA_COMPILER "${_CUDA118}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
endif()

# ── MSVC host compiler for CUDA 11.8 (must be < 14.40) ────────────────────────
set(_MSVC_ROOT "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC")
set(_CUDA_HOST_CL "")
if(EXISTS "${_MSVC_ROOT}")
    file(GLOB _MSVC_DIRS LIST_DIRECTORIES true "${_MSVC_ROOT}/14.*")
    set(_BEST_VER "")
    foreach(_DIR ${_MSVC_DIRS})
        get_filename_component(_VER "${_DIR}" NAME)
        if(_VER VERSION_LESS "14.40" AND EXISTS "${_DIR}/bin/Hostx64/x64/cl.exe")
            if(_BEST_VER STREQUAL "" OR _VER VERSION_GREATER _BEST_VER)
                set(_BEST_VER "${_VER}")
                set(_CUDA_HOST_CL "${_DIR}/bin/Hostx64/x64/cl.exe")
                set(_MSVC_LIB "${_DIR}/lib/x64")
            endif()
        endif()
    endforeach()
endif()

if(_CUDA_HOST_CL)
    message(STATUS "CUDA 11.8 host compiler: ${_CUDA_HOST_CL}")
    # Set CXX compiler so cmake uses 14.39 for C++ files.
    # Do NOT set CMAKE_CUDA_HOST_COMPILER: that makes cmake pass -ccbin to nvcc,
    # which then conflicts with the cl.exe nvcc finds in PATH via --use-local-env.
    # Instead, nvcc picks up cl.exe from PATH (set by setup_cuda118_env.bat).
    set(CMAKE_CXX_COMPILER "${_CUDA_HOST_CL}" CACHE FILEPATH "" FORCE)
else()
    message(WARNING "No MSVC < 14.40 found. Install MSVC v14.39 via Visual Studio Installer.")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --use-local-env"
    CACHE STRING "Flags for CUDA compilation" FORCE)

project(lbm_cuda LANGUAGES CXX CUDA)

# Ninja: main.exe ends up in the build/ folder (not build/Release/).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(SDL3 REQUIRED)

add_executable(main main.cu)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

target_link_libraries(main PRIVATE SDL3::SDL3)

if(WIN32)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL3::SDL3>
            $<TARGET_FILE_DIR:main>
        COMMENT "Copying SDL3.dll to output directory"
    )
endif()
