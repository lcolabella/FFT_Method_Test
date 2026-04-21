cmake_minimum_required(VERSION 3.16)
project(FFT_Method VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FFT_METHOD_ENABLE_OPENMP "Enable OpenMP parallel execution" ON)
option(FFT_METHOD_ENABLE_TESTS "Build tests" ON)

set(FFT_METHOD_COMMON_SOURCES
    src/geometry/BinaryMedium.cpp
    src/geometry/InterfaceVoxels.cpp
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

add_executable(fft_method
    src/main.cpp
    src/app/SimulationConfig.cpp
    src/app/PermeabilityRunner.cpp
    ${FFT_METHOD_COMMON_SOURCES}
)

target_include_directories(fft_method PRIVATE src)

if(MSVC)
    target_compile_options(fft_method PRIVATE /W4)
else()
    target_compile_options(fft_method PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(FFT_METHOD_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fft_method PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(fft_method PRIVATE PERMEABILITY_USE_OPENMP=1)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(fft_method PRIVATE Threads::Threads)

find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3)
find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads)
if(FFTW3_LIBRARY)
    target_link_libraries(fft_method PRIVATE ${FFTW3_LIBRARY})
    if(FFTW3_INCLUDE_DIR)
        target_include_directories(fft_method PRIVATE ${FFTW3_INCLUDE_DIR})
    endif()
    target_compile_definitions(fft_method PRIVATE PERMEABILITY_HAVE_FFTW3=1)
    if(FFTW3_THREADS_LIBRARY)
        target_link_libraries(fft_method PRIVATE ${FFTW3_THREADS_LIBRARY})
    endif()
endif()

if(FFT_METHOD_ENABLE_TESTS)
    enable_testing()

    function(add_fft_method_test target_name test_source)
        add_executable(${target_name}
            ${test_source}
            ${FFT_METHOD_COMMON_SOURCES}
        )
        target_include_directories(${target_name} PRIVATE src)
        target_link_libraries(${target_name} PRIVATE Threads::Threads)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(FFTW3_LIBRARY)
            target_link_libraries(${target_name} PRIVATE ${FFTW3_LIBRARY})
        endif()
        if(FFTW3_THREADS_LIBRARY)
            target_link_libraries(${target_name} PRIVATE ${FFTW3_THREADS_LIBRARY})
        endif()
        if(FFTW3_INCLUDE_DIR)
            target_include_directories(${target_name} PRIVATE ${FFTW3_INCLUDE_DIR})
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
endif()
