#include "fft/SYCLFFTBackend.hpp"

#if defined(PERMEABILITY_USE_SYCL)

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <oneapi/mkl/dft.hpp>
#include <sycl/sycl.hpp>

namespace permeability {

namespace {

#if defined(FFT_METHOD_PRECISION_DOUBLE)
using DeviceComplex = std::complex<double>;
using DftDescriptor = oneapi::mkl::dft::descriptor<
    oneapi::mkl::dft::precision::DOUBLE,
    oneapi::mkl::dft::domain::COMPLEX>;
#else
using DeviceComplex = std::complex<float>;
using DftDescriptor = oneapi::mkl::dft::descriptor<
    oneapi::mkl::dft::precision::SINGLE,
    oneapi::mkl::dft::domain::COMPLEX>;
#endif

void ensure_same_size(std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != rhs) {
        throw std::invalid_argument(message);
    }
}

}  // namespace

struct SYCLFFTBackend::Impl {
    Grid3D grid{1, 1, 1};
    bool initialized = false;
    double inv_n = 1.0;
    std::string device_name;

    sycl::queue queue;
    DftDescriptor descriptor;

    DeviceComplex* d_in = nullptr;
    DeviceComplex* d_out = nullptr;
    DeviceComplex* d_force_x = nullptr;
    DeviceComplex* d_force_y = nullptr;
    DeviceComplex* d_force_z = nullptr;
    DeviceComplex* d_vel_x = nullptr;
    DeviceComplex* d_vel_y = nullptr;
    DeviceComplex* d_vel_z = nullptr;
    Scalar* d_gxx = nullptr;
    Scalar* d_gxy = nullptr;
    Scalar* d_gxz = nullptr;
    Scalar* d_gyy = nullptr;
    Scalar* d_gyz = nullptr;
    Scalar* d_gzz = nullptr;

    std::vector<DeviceComplex> host_work;

    static void sycl_async_error_handler(sycl::exception_list exceptions) {
        for (const std::exception_ptr& ep : exceptions) {
            try {
                std::rethrow_exception(ep);
            } catch (const sycl::exception& e) {
                throw std::runtime_error(
                    std::string("SYCLFFTBackend async SYCL error: ") + e.what());
            }
        }
    }

    Impl()
        : queue(sycl::gpu_selector_v,
                sycl_async_error_handler,
                sycl::property::queue::in_order{}),
          descriptor(static_cast<std::int64_t>(1)) {
        device_name = queue.get_device().get_info<sycl::info::device::name>();
    }

    ~Impl() {
        auto release = [&](auto*& p) {
            if (p != nullptr) {
                sycl::free(p, queue);
                p = nullptr;
            }
        };

        release(d_in);
        release(d_out);
        release(d_force_x);
        release(d_force_y);
        release(d_force_z);
        release(d_vel_x);
        release(d_vel_y);
        release(d_vel_z);
        release(d_gxx);
        release(d_gxy);
        release(d_gxz);
        release(d_gyy);
        release(d_gyz);
        release(d_gzz);
    }
};

SYCLFFTBackend::SYCLFFTBackend() : impl_(std::make_unique<Impl>()) {}
SYCLFFTBackend::~SYCLFFTBackend() = default;
SYCLFFTBackend::SYCLFFTBackend(SYCLFFTBackend&&) noexcept = default;
SYCLFFTBackend& SYCLFFTBackend::operator=(SYCLFFTBackend&&) noexcept = default;

void SYCLFFTBackend::initialize(const Grid3D& grid, int /*thread_count*/) {
    impl_->grid = grid;
    const std::size_t n = grid.total_size();
    impl_->inv_n = 1.0 / static_cast<double>(n);
    impl_->host_work.assign(n, DeviceComplex{});

    auto alloc_complex = [&](DeviceComplex*& p) {
        if (p != nullptr) {
            sycl::free(p, impl_->queue);
            p = nullptr;
        }
        p = sycl::malloc_device<DeviceComplex>(n, impl_->queue);
        if (p == nullptr) {
            throw std::runtime_error("SYCLFFTBackend allocation failed (complex buffer)");
        }
    };

    auto alloc_scalar = [&](Scalar*& p) {
        if (p != nullptr) {
            sycl::free(p, impl_->queue);
            p = nullptr;
        }
        p = sycl::malloc_device<Scalar>(n, impl_->queue);
        if (p == nullptr) {
            throw std::runtime_error("SYCLFFTBackend allocation failed (scalar buffer)");
        }
    };

    alloc_complex(impl_->d_in);
    alloc_complex(impl_->d_out);
    alloc_complex(impl_->d_force_x);
    alloc_complex(impl_->d_force_y);
    alloc_complex(impl_->d_force_z);
    alloc_complex(impl_->d_vel_x);
    alloc_complex(impl_->d_vel_y);
    alloc_complex(impl_->d_vel_z);
    alloc_scalar(impl_->d_gxx);
    alloc_scalar(impl_->d_gxy);
    alloc_scalar(impl_->d_gxz);
    alloc_scalar(impl_->d_gyy);
    alloc_scalar(impl_->d_gyz);
    alloc_scalar(impl_->d_gzz);

    const std::vector<std::int64_t> lengths = {
        static_cast<std::int64_t>(grid.nz()),
        static_cast<std::int64_t>(grid.ny()),
        static_cast<std::int64_t>(grid.nx()),
    };
    impl_->descriptor = DftDescriptor(lengths);
    impl_->descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                oneapi::mkl::dft::config_value::NOT_INPLACE);
    impl_->descriptor.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                                static_cast<Scalar>(impl_->inv_n));
    impl_->descriptor.commit(impl_->queue);

    impl_->initialized = true;
}

void SYCLFFTBackend::forward(const VectorField3D& real_in,
                             ComplexVectorField3D& complex_out) {
    if (!impl_->initialized) {
        throw std::runtime_error("SYCLFFTBackend must be initialized before forward transform");
    }
    ensure_same_size(real_in.size(), complex_out.size(), "SYCLFFTBackend::forward size mismatch");

    const std::size_t n = real_in.size();
    auto run_component = [&](const ScalarField3D& src, Array3D<ScalarComplex>& dst) {
        for (std::size_t i = 0; i < n; ++i) {
            impl_->host_work[i] = DeviceComplex(static_cast<decltype(DeviceComplex{}.real())>(src[i]),
                                                static_cast<decltype(DeviceComplex{}.imag())>(0));
        }

        impl_->queue.memcpy(impl_->d_in, impl_->host_work.data(), n * sizeof(DeviceComplex)).wait();
        oneapi::mkl::dft::compute_forward(impl_->descriptor, impl_->d_in, impl_->d_out).wait();
        impl_->queue.memcpy(impl_->host_work.data(), impl_->d_out, n * sizeof(DeviceComplex)).wait();

        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = ScalarComplex(static_cast<Scalar>(impl_->host_work[i].real()),
                                   static_cast<Scalar>(impl_->host_work[i].imag()));
        }
    };

    run_component(real_in.x(), complex_out.x());
    run_component(real_in.y(), complex_out.y());
    run_component(real_in.z(), complex_out.z());
}

void SYCLFFTBackend::inverse(const ComplexVectorField3D& complex_in,
                             VectorField3D& real_out) {
    if (!impl_->initialized) {
        throw std::runtime_error("SYCLFFTBackend must be initialized before inverse transform");
    }
    ensure_same_size(real_out.size(), complex_in.size(), "SYCLFFTBackend::inverse size mismatch");

    const std::size_t n = real_out.size();
    auto run_component = [&](const Array3D<ScalarComplex>& src, ScalarField3D& dst) {
        for (std::size_t i = 0; i < n; ++i) {
            impl_->host_work[i] = DeviceComplex(
                static_cast<decltype(DeviceComplex{}.real())>(src[i].real()),
                static_cast<decltype(DeviceComplex{}.imag())>(src[i].imag()));
        }

        impl_->queue.memcpy(impl_->d_in, impl_->host_work.data(), n * sizeof(DeviceComplex)).wait();
        oneapi::mkl::dft::compute_backward(impl_->descriptor, impl_->d_in, impl_->d_out).wait();
        impl_->queue.memcpy(impl_->host_work.data(), impl_->d_out, n * sizeof(DeviceComplex)).wait();

        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<Scalar>(impl_->host_work[i].real());
        }
    };

    run_component(complex_in.x(), real_out.x());
    run_component(complex_in.y(), real_out.y());
    run_component(complex_in.z(), real_out.z());
}

bool SYCLFFTBackend::apply_spectral_green_tensor(const std::vector<Scalar>& spectral_tensor_xx,
                                                 const std::vector<Scalar>& spectral_tensor_xy,
                                                 const std::vector<Scalar>& spectral_tensor_xz,
                                                 const std::vector<Scalar>& spectral_tensor_yy,
                                                 const std::vector<Scalar>& spectral_tensor_yz,
                                                 const std::vector<Scalar>& spectral_tensor_zz,
                                                 const ComplexVectorField3D& spectral_force,
                                                 ComplexVectorField3D& spectral_velocity) {
    if (!impl_->initialized) {
        throw std::runtime_error("SYCLFFTBackend must be initialized before Green tensor application");
    }

    const std::size_t n = spectral_force.size();
    ensure_same_size(n, spectral_velocity.size(), "SYCLFFTBackend::apply_spectral_green_tensor size mismatch");
    ensure_same_size(n, spectral_tensor_xx.size(), "SYCLFFTBackend::apply_spectral_green_tensor tensor size mismatch");
    ensure_same_size(n, spectral_tensor_xy.size(), "SYCLFFTBackend::apply_spectral_green_tensor tensor size mismatch");
    ensure_same_size(n, spectral_tensor_xz.size(), "SYCLFFTBackend::apply_spectral_green_tensor tensor size mismatch");
    ensure_same_size(n, spectral_tensor_yy.size(), "SYCLFFTBackend::apply_spectral_green_tensor tensor size mismatch");
    ensure_same_size(n, spectral_tensor_yz.size(), "SYCLFFTBackend::apply_spectral_green_tensor tensor size mismatch");
    ensure_same_size(n, spectral_tensor_zz.size(), "SYCLFFTBackend::apply_spectral_green_tensor tensor size mismatch");

    impl_->queue.memcpy(impl_->d_gxx, spectral_tensor_xx.data(), n * sizeof(Scalar));
    impl_->queue.memcpy(impl_->d_gxy, spectral_tensor_xy.data(), n * sizeof(Scalar));
    impl_->queue.memcpy(impl_->d_gxz, spectral_tensor_xz.data(), n * sizeof(Scalar));
    impl_->queue.memcpy(impl_->d_gyy, spectral_tensor_yy.data(), n * sizeof(Scalar));
    impl_->queue.memcpy(impl_->d_gyz, spectral_tensor_yz.data(), n * sizeof(Scalar));
    impl_->queue.memcpy(impl_->d_gzz, spectral_tensor_zz.data(), n * sizeof(Scalar));

    impl_->queue.memcpy(impl_->d_force_x, spectral_force.x().data(), n * sizeof(ScalarComplex));
    impl_->queue.memcpy(impl_->d_force_y, spectral_force.y().data(), n * sizeof(ScalarComplex));
    impl_->queue.memcpy(impl_->d_force_z, spectral_force.z().data(), n * sizeof(ScalarComplex));

    impl_->queue.submit([&](sycl::handler& h) {
        auto* gxx = impl_->d_gxx;
        auto* gxy = impl_->d_gxy;
        auto* gxz = impl_->d_gxz;
        auto* gyy = impl_->d_gyy;
        auto* gyz = impl_->d_gyz;
        auto* gzz = impl_->d_gzz;
        auto* fx = impl_->d_force_x;
        auto* fy = impl_->d_force_y;
        auto* fz = impl_->d_force_z;
        auto* vx = impl_->d_vel_x;
        auto* vy = impl_->d_vel_y;
        auto* vz = impl_->d_vel_z;

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const std::size_t i = id[0];
            const ScalarComplex fx_i = fx[i];
            const ScalarComplex fy_i = fy[i];
            const ScalarComplex fz_i = fz[i];

            vx[i] = gxx[i] * fx_i + gxy[i] * fy_i + gxz[i] * fz_i;
            vy[i] = gxy[i] * fx_i + gyy[i] * fy_i + gyz[i] * fz_i;
            vz[i] = gxz[i] * fx_i + gyz[i] * fy_i + gzz[i] * fz_i;
        });
    }).wait();

    impl_->queue.memcpy(spectral_velocity.x().data(), impl_->d_vel_x, n * sizeof(ScalarComplex));
    impl_->queue.memcpy(spectral_velocity.y().data(), impl_->d_vel_y, n * sizeof(ScalarComplex));
    impl_->queue.memcpy(spectral_velocity.z().data(), impl_->d_vel_z, n * sizeof(ScalarComplex)).wait();

    return true;
}

double SYCLFFTBackend::normalization_factor() const noexcept {
    return impl_->inv_n;
}

const Grid3D& SYCLFFTBackend::grid() const noexcept {
    return impl_->grid;
}

std::string SYCLFFTBackend::device_name() const {
    return impl_->device_name;
}

std::string SYCLFFTBackend::device_info() const {
    const auto& dev = impl_->queue.get_device();
    const auto max_cu = dev.get_info<sycl::info::device::max_compute_units>();
    return "SYCL GPU: " + impl_->device_name +
           " (" + std::to_string(max_cu) + " CUs)";
}

}  // namespace permeability

#endif  // PERMEABILITY_USE_SYCL
