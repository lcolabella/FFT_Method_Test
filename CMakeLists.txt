cmake_minimum_required(VERSION 3.16)
project(FFT_Method VERSION 0.4.0 LANGUAGES CXX)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FFT_METHOD_ENABLE_OPENMP "Enable OpenMP parallel execution" ON)
option(FFT_METHOD_ENABLE_TESTS "Build tests" ON)
option(FFT_METHOD_STRICT_FP "Use strict floating-point mode for improved numerical reproducibility" OFF)

# Phase 5: Configurable precision and FFT backend
set(FFT_METHOD_PRECISION "FLOAT" CACHE STRING "Scalar precision: FLOAT or DOUBLE (default: FLOAT)")
set_property(CACHE FFT_METHOD_PRECISION PROPERTY STRINGS "FLOAT" "DOUBLE")

set(FFT_METHOD_FFT_BACKEND "AUTO" CACHE STRING "FFT backend: AUTO, FFTW3F, or FFTW3 (default: AUTO)")
set_property(CACHE FFT_METHOD_FFT_BACKEND PROPERTY STRINGS "AUTO" "FFTW3F" "FFTW3")

set(FFT_METHOD_GPU_BACKEND "NONE" CACHE STRING "GPU backend: NONE or SYCL (default: NONE)")
set_property(CACHE FFT_METHOD_GPU_BACKEND PROPERTY STRINGS "NONE" "SYCL")

set(FFT_METHOD_SYCL_TARGET "AUTO" CACHE STRING "SYCL target: AUTO, INTEL_GPU, NVIDIA_CUDA, SPIR64")
set_property(CACHE FFT_METHOD_SYCL_TARGET PROPERTY STRINGS "AUTO" "INTEL_GPU" "NVIDIA_CUDA" "SPIR64")

set(FFT_METHOD_CUDA_PATH "" CACHE PATH "Path to CUDA toolkit root (e.g. /usr/local/cuda); required for NVIDIA_CUDA target")

# Validate precision
if(NOT FFT_METHOD_PRECISION MATCHES "^(FLOAT|DOUBLE)$")
    message(FATAL_ERROR "FFT_METHOD_PRECISION must be FLOAT or DOUBLE, got: ${FFT_METHOD_PRECISION}")
endif()

# Validate backend
if(NOT FFT_METHOD_FFT_BACKEND MATCHES "^(AUTO|FFTW3F|FFTW3)$")
    message(FATAL_ERROR "FFT_METHOD_FFT_BACKEND must be AUTO, FFTW3F, or FFTW3, got: ${FFT_METHOD_FFT_BACKEND}")
endif()

# Validate GPU backend / SYCL target
if(NOT FFT_METHOD_GPU_BACKEND MATCHES "^(NONE|SYCL)$")
    message(FATAL_ERROR "FFT_METHOD_GPU_BACKEND must be NONE or SYCL, got: ${FFT_METHOD_GPU_BACKEND}")
endif()
if(NOT FFT_METHOD_SYCL_TARGET MATCHES "^(AUTO|INTEL_GPU|NVIDIA_CUDA|SPIR64)$")
    message(FATAL_ERROR "FFT_METHOD_SYCL_TARGET must be AUTO, INTEL_GPU, NVIDIA_CUDA, or SPIR64, got: ${FFT_METHOD_SYCL_TARGET}")
endif()

message(STATUS "FFT_METHOD_PRECISION: ${FFT_METHOD_PRECISION}")
message(STATUS "FFT_METHOD_FFT_BACKEND: ${FFT_METHOD_FFT_BACKEND}")
message(STATUS "FFT_METHOD_GPU_BACKEND: ${FFT_METHOD_GPU_BACKEND}")
message(STATUS "FFT_METHOD_SYCL_TARGET: ${FFT_METHOD_SYCL_TARGET}")
message(STATUS "FFT_METHOD_STRICT_FP: ${FFT_METHOD_STRICT_FP}")

set(FFT_METHOD_SYCL_FLAGS)
if(FFT_METHOD_GPU_BACKEND STREQUAL "SYCL")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(IntelLLVM|Clang)$")
        message(FATAL_ERROR
            "FFT_METHOD_GPU_BACKEND=SYCL requires a SYCL-capable compiler (for example icpx). "
            "Current compiler is ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}).")
    endif()

    check_cxx_compiler_flag("-fsycl" FFT_METHOD_HAS_FSYCL_FLAG)
    if(NOT FFT_METHOD_HAS_FSYCL_FLAG)
        message(FATAL_ERROR
            "Compiler ${CMAKE_CXX_COMPILER} does not support -fsycl. "
            "Use a SYCL-enabled compiler (for example icpx) and reconfigure.")
    endif()

    list(APPEND FFT_METHOD_SYCL_FLAGS -fsycl)

    if(FFT_METHOD_SYCL_TARGET STREQUAL "INTEL_GPU")
        list(APPEND FFT_METHOD_SYCL_FLAGS -fsycl-targets=spir64)
    elseif(FFT_METHOD_SYCL_TARGET STREQUAL "NVIDIA_CUDA")
        list(APPEND FFT_METHOD_SYCL_FLAGS -fsycl-targets=nvptx64-nvidia-cuda)

        # Resolve CUDA path: use explicit option, then common install locations.
        set(_cuda_path "${FFT_METHOD_CUDA_PATH}")
        if(NOT _cuda_path)
            foreach(_candidate "/usr/local/cuda" "/usr/cuda" "/opt/cuda")
                if(EXISTS "${_candidate}/nvvm/libdevice/libdevice.10.bc")
                    set(_cuda_path "${_candidate}")
                    break()
                endif()
            endforeach()
        endif()

        if(_cuda_path)
            message(STATUS "CUDA toolkit path: ${_cuda_path}")
            list(APPEND FFT_METHOD_SYCL_FLAGS "--cuda-path=${_cuda_path}")
        else()
            message(WARNING
                "CUDA toolkit not found at common paths. "
                "If you get 'cannot find libdevice' errors, set "
                "-DFFT_METHOD_CUDA_PATH=/path/to/cuda when configuring.")
        endif()
        unset(_cuda_path)
    elseif(FFT_METHOD_SYCL_TARGET STREQUAL "SPIR64")
        list(APPEND FFT_METHOD_SYCL_FLAGS -fsycl-targets=spir64)
    endif()

    add_compile_definitions(PERMEABILITY_USE_SYCL=1)
    add_compile_options(${FFT_METHOD_SYCL_FLAGS})
    add_link_options(${FFT_METHOD_SYCL_FLAGS})

    message(STATUS "SYCL compile/link flags: ${FFT_METHOD_SYCL_FLAGS}")

    # Locate oneMKL for SYCL DFT.  Intel oneAPI installs a MKLConfig.cmake
    # under $MKLROOT/lib/cmake/mkl.  We try that hint before the system paths.
    find_package(MKL CONFIG
        HINTS "$ENV{MKLROOT}/lib/cmake/mkl"
              "$ENV{MKLROOT}/lib/cmake"
              "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl"
    )
    if(MKL_FOUND)
        message(STATUS "Found oneMKL: ${MKL_VERSION}")
        set(FFT_METHOD_ONEMKL_FOUND TRUE)
    else()
        # Fallback: check for the oneMKL standalone package (open-source version)
        find_package(oneMKL CONFIG
            HINTS "$ENV{ONEMKL_DIR}"
                  "/opt/intel/oneapi/mkl/latest"
        )
        if(oneMKL_FOUND)
            message(STATUS "Found oneMKL (open-source): ${oneMKL_VERSION}")
            set(FFT_METHOD_ONEMKL_FOUND TRUE)
        else()
            message(WARNING
                "oneMKL CMake config not found (tried MKL and oneMKL packages). "
                "The SYCL DFT backend will compile but may fail to link. "
                "Set MKLROOT or ONEMKL_DIR, or add oneMKL libs to CMAKE_PREFIX_PATH.")
            set(FFT_METHOD_ONEMKL_FOUND FALSE)
        endif()
    endif()
endif()

set(FFT_METHOD_COMMON_SOURCES
    src/logging/Logger.cpp
    src/materials/MaterialDatabase.cpp
    src/brinkman/FFT.cpp
    src/brinkman/FluidBrinkman.cpp
    src/geometry/BinaryMedium.cpp
    src/geometry/InterfaceVoxels.cpp
    src/fft/FFTBackendFactory.cpp
    src/fft/FFTWBackend.cpp
    src/io/VoxelIO.cpp
    src/bignonnet/WaveVectorGrid.cpp
    src/bignonnet/ForceSupport.cpp
    src/bignonnet/TrialForceField.cpp
    src/bignonnet/BackgroundForce.cpp
    src/bignonnet/GreenOperator.cpp
    src/bignonnet/VariationalOperator.cpp
    src/bignonnet/VariationalRHS.cpp
    src/bignonnet/VelocityRecovery.cpp
    src/solver/MinresSolver.cpp
    src/post/PermeabilityTensor.cpp
    src/post/Diagnostics.cpp
)

if(FFT_METHOD_GPU_BACKEND STREQUAL "SYCL")
    list(APPEND FFT_METHOD_COMMON_SOURCES src/fft/SYCLFFTBackend.cpp)
endif()

add_executable(fft_method
    src/main.cpp
    src/app/SimulationConfig.cpp
    src/app/PermeabilityRunner.cpp
    ${FFT_METHOD_COMMON_SOURCES}
)

target_include_directories(fft_method PRIVATE src include)

if(MSVC)
    target_compile_options(fft_method PRIVATE /W4)
else()
    target_compile_options(fft_method PRIVATE -Wall -Wextra -Wpedantic)
    if(FFT_METHOD_STRICT_FP)
        target_compile_options(fft_method PRIVATE -ffp-model=precise)
    endif()
endif()

if(FFT_METHOD_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fft_method PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(fft_method PRIVATE PERMEABILITY_USE_OPENMP=1)
        target_compile_definitions(fft_method PRIVATE COMMON_USE_OPENMP=1)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(fft_method PRIVATE Threads::Threads)

find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3)
find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads)
find_library(FFTW3F_LIBRARY NAMES fftw3f)
find_library(FFTW3F_THREADS_LIBRARY NAMES fftw3f_threads)

# Phase 5: Backend selection logic
set(HAVE_FFTW3F FALSE)
set(HAVE_FFTW3 FALSE)
set(PREFERRED_BACKEND "NONE")

if(FFTW3F_LIBRARY)
    set(HAVE_FFTW3F TRUE)
    if(FFT_METHOD_FFT_BACKEND STREQUAL "AUTO" OR FFT_METHOD_FFT_BACKEND STREQUAL "FFTW3F")
        set(PREFERRED_BACKEND "FFTW3F")
    endif()
endif()

if(FFTW3_LIBRARY AND PREFERRED_BACKEND STREQUAL "NONE")
    set(HAVE_FFTW3 TRUE)
    if(FFT_METHOD_FFT_BACKEND STREQUAL "AUTO" OR FFT_METHOD_FFT_BACKEND STREQUAL "FFTW3")
        set(PREFERRED_BACKEND "FFTW3")
    endif()
endif()

# Override with explicit backend request
if(FFT_METHOD_FFT_BACKEND STREQUAL "FFTW3F")
    if(NOT FFTW3F_LIBRARY)
        message(FATAL_ERROR "FFT_METHOD_FFT_BACKEND=FFTW3F requested but libfftw3f not found")
    endif()
    set(PREFERRED_BACKEND "FFTW3F")
elseif(FFT_METHOD_FFT_BACKEND STREQUAL "FFTW3")
    if(NOT FFTW3_LIBRARY)
        message(FATAL_ERROR "FFT_METHOD_FFT_BACKEND=FFTW3 requested but libfftw3 not found")
    endif()
    set(PREFERRED_BACKEND "FFTW3")
endif()

if(PREFERRED_BACKEND STREQUAL "NONE")
    if(FFT_METHOD_GPU_BACKEND STREQUAL "SYCL")
        message(WARNING
            "No FFTW library found. Building GPU-only SYCL binary. "
            "CPU backend will be unavailable at runtime.")
    else()
        message(FATAL_ERROR "No FFTW library found. Install fftw3 or fftw3f")
    endif()
endif()

if(PREFERRED_BACKEND STREQUAL "NONE")
    message(STATUS "Using FFT backend: NONE (GPU-only build)")
else()
    message(STATUS "Using FFT backend: ${PREFERRED_BACKEND}")
endif()

# Link the preferred backend
if(PREFERRED_BACKEND STREQUAL "FFTW3F")
    target_link_libraries(fft_method PRIVATE ${FFTW3F_LIBRARY})
    target_compile_definitions(fft_method PRIVATE PERMEABILITY_HAVE_FFTW3F=1)
    if(FFTW3F_THREADS_LIBRARY)
        target_link_libraries(fft_method PRIVATE ${FFTW3F_THREADS_LIBRARY})
    endif()
elseif(PREFERRED_BACKEND STREQUAL "FFTW3")
    target_link_libraries(fft_method PRIVATE ${FFTW3_LIBRARY})
    target_compile_definitions(fft_method PRIVATE PERMEABILITY_HAVE_FFTW3=1)
    target_compile_definitions(fft_method PRIVATE COMMON_USE_FFTW=1)
    if(FFTW3_THREADS_LIBRARY)
        target_link_libraries(fft_method PRIVATE ${FFTW3_THREADS_LIBRARY})
    endif()
endif()

if(FFTW3_INCLUDE_DIR)
    target_include_directories(fft_method PRIVATE ${FFTW3_INCLUDE_DIR})
endif()

# Set precision define
if(FFT_METHOD_PRECISION STREQUAL "FLOAT")
    target_compile_definitions(fft_method PRIVATE FFT_METHOD_PRECISION_FLOAT=1)
else()
    target_compile_definitions(fft_method PRIVATE FFT_METHOD_PRECISION_DOUBLE=1)
endif()

# Link oneMKL for the SYCL GPU backend
if(FFT_METHOD_GPU_BACKEND STREQUAL "SYCL" AND FFT_METHOD_ONEMKL_FOUND)
    if(MKL_FOUND)
        # Intel oneAPI MKL – SYCL interface target
        target_link_libraries(fft_method PRIVATE MKL::MKL_SYCL)
    elseif(oneMKL_FOUND)
        target_link_libraries(fft_method PRIVATE oneMKL::oneMKL)
    endif()
endif()

if(FFT_METHOD_ENABLE_TESTS AND NOT PREFERRED_BACKEND STREQUAL "NONE")
    enable_testing()

    function(add_fft_method_test target_name test_source)
        add_executable(${target_name}
            ${test_source}
            ${FFT_METHOD_COMMON_SOURCES}
        )
        target_include_directories(${target_name} PRIVATE src include)
        target_link_libraries(${target_name} PRIVATE Threads::Threads)
        if(FFT_METHOD_STRICT_FP AND NOT MSVC)
            target_compile_options(${target_name} PRIVATE -ffp-model=precise)
        endif()
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        
        # Apply same backend selection as main target
        if(PREFERRED_BACKEND STREQUAL "FFTW3F")
            target_link_libraries(${target_name} PRIVATE ${FFTW3F_LIBRARY})
            target_compile_definitions(${target_name} PRIVATE PERMEABILITY_HAVE_FFTW3F=1)
            if(FFTW3F_THREADS_LIBRARY)
                target_link_libraries(${target_name} PRIVATE ${FFTW3F_THREADS_LIBRARY})
            endif()
        elseif(PREFERRED_BACKEND STREQUAL "FFTW3")
            target_link_libraries(${target_name} PRIVATE ${FFTW3_LIBRARY})
            target_compile_definitions(${target_name} PRIVATE PERMEABILITY_HAVE_FFTW3=1)
            if(FFTW3_THREADS_LIBRARY)
                target_link_libraries(${target_name} PRIVATE ${FFTW3_THREADS_LIBRARY})
            endif()
        endif()
        
        if(FFTW3_INCLUDE_DIR)
            target_include_directories(${target_name} PRIVATE ${FFTW3_INCLUDE_DIR})
        endif()
        
        # Apply precision setting
        if(FFT_METHOD_PRECISION STREQUAL "FLOAT")
            target_compile_definitions(${target_name} PRIVATE FFT_METHOD_PRECISION_FLOAT=1)
        else()
            target_compile_definitions(${target_name} PRIVATE FFT_METHOD_PRECISION_DOUBLE=1)
        endif()

        # Link oneMKL for SYCL GPU backend
        if(FFT_METHOD_GPU_BACKEND STREQUAL "SYCL" AND FFT_METHOD_ONEMKL_FOUND)
            if(MKL_FOUND)
                target_link_libraries(${target_name} PRIVATE MKL::MKL_SYCL)
            elseif(oneMKL_FOUND)
                target_link_libraries(${target_name} PRIVATE oneMKL::oneMKL)
            endif()
        endif()
        
        add_test(NAME ${target_name} COMMAND ${target_name})
    endfunction()

    add_fft_method_test(test_grid tests/test_grid.cpp)
    add_fft_method_test(test_interface_voxels tests/test_interface_voxels.cpp)
    add_fft_method_test(test_support_mapping tests/test_support_mapping.cpp)
    add_fft_method_test(test_fft_roundtrip tests/test_fft_roundtrip.cpp)
    add_fft_method_test(test_background_force tests/test_background_force.cpp)

    add_fft_method_test(test_projector_green tests/test_projector_green.cpp)
    add_fft_method_test(test_variational_rhs tests/test_variational_rhs.cpp)
    add_fft_method_test(test_variational_operator_symmetry tests/test_variational_operator_symmetry.cpp)
    add_fft_method_test(test_velocity_recovery tests/test_velocity_recovery.cpp)
    add_fft_method_test(test_small_permeability_pipeline tests/test_small_permeability_pipeline.cpp)
    add_fft_method_test(test_voxelio_load_geometry tests/test_voxelio_load_geometry.cpp)
    add_fft_method_test(test_material_database tests/test_material_database.cpp)

    add_executable(test_simulation_config
        tests/test_simulation_config.cpp
        src/app/SimulationConfig.cpp
        ${FFT_METHOD_COMMON_SOURCES}
    )
    target_include_directories(test_simulation_config PRIVATE src include)
    target_link_libraries(test_simulation_config PRIVATE Threads::Threads)
    if(FFT_METHOD_STRICT_FP AND NOT MSVC)
        target_compile_options(test_simulation_config PRIVATE -ffp-model=precise)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_simulation_config PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Apply same backend selection as main target
    if(PREFERRED_BACKEND STREQUAL "FFTW3F")
        target_link_libraries(test_simulation_config PRIVATE ${FFTW3F_LIBRARY})
        target_compile_definitions(test_simulation_config PRIVATE PERMEABILITY_HAVE_FFTW3F=1)
        if(FFTW3F_THREADS_LIBRARY)
            target_link_libraries(test_simulation_config PRIVATE ${FFTW3F_THREADS_LIBRARY})
        endif()
    elseif(PREFERRED_BACKEND STREQUAL "FFTW3")
        target_link_libraries(test_simulation_config PRIVATE ${FFTW3_LIBRARY})
        target_compile_definitions(test_simulation_config PRIVATE PERMEABILITY_HAVE_FFTW3=1)
        if(FFTW3_THREADS_LIBRARY)
            target_link_libraries(test_simulation_config PRIVATE ${FFTW3_THREADS_LIBRARY})
        endif()
    endif()
    
    if(FFTW3_INCLUDE_DIR)
        target_include_directories(test_simulation_config PRIVATE ${FFTW3_INCLUDE_DIR})
    endif()
    
    # Apply precision setting
    if(FFT_METHOD_PRECISION STREQUAL "FLOAT")
        target_compile_definitions(test_simulation_config PRIVATE FFT_METHOD_PRECISION_FLOAT=1)
    else()
        target_compile_definitions(test_simulation_config PRIVATE FFT_METHOD_PRECISION_DOUBLE=1)
    endif()

    # Link oneMKL for SYCL GPU backend
    if(FFT_METHOD_GPU_BACKEND STREQUAL "SYCL" AND FFT_METHOD_ONEMKL_FOUND)
        if(MKL_FOUND)
            target_link_libraries(test_simulation_config PRIVATE MKL::MKL_SYCL)
        elseif(oneMKL_FOUND)
            target_link_libraries(test_simulation_config PRIVATE oneMKL::oneMKL)
        endif()
    endif()
    
    add_test(NAME test_simulation_config COMMAND test_simulation_config)
elseif(FFT_METHOD_ENABLE_TESTS)
    message(WARNING
        "FFT_METHOD_ENABLE_TESTS=ON but no FFTW backend is available. "
        "Skipping tests in GPU-only build.")
endif()
