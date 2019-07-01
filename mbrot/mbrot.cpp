#include "mbrot.h"
#include <complex>

void run_in_gpu(png_byte *data, std::complex<float> c0, std::complex<float> c1,
                std::complex<float> delta, size_t lo, size_t hi, size_t width,
                size_t height, size_t threads) {
    thrust::complex<float> c = c0 + std::complex<float>{0., delta.imag() * lo};

    gpu::array<png_byte> gpu_data(4 * width * (hi - lo));
    int num_blocks = (width * (hi - lo) + threads) / threads;
    calculate_indices<10000><<<num_blocks, threads>>>(
        width, hi - lo, c, delta.real(), delta.imag(), gpu_data.data());

    gpu_data.get(data);
}

void run_in_cpu(png_byte *data, std::complex<float> c0, std::complex<float> c1,
                std::complex<float> delta, size_t lo, size_t hi, size_t width,
                size_t height, size_t threads) {
    using C = thrust::complex<float>;
    C c[2] = {{c0.real(), c0.imag()}, {c1.real(), c1.imag()}};

    omp_set_num_threads(threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t y = lo; y < hi; y++)
        for (size_t x = 0; x < width; x++) {
            const size_t idx = width * (y - lo) + x;
            auto &pixel = reinterpret_cast<pic::canvas::rgba_reference &>(
                data[4 * idx]);
            pic::colorize(
                pixel,
                iterate<10000>(c[0] + C{x * delta.real(), y * delta.imag()}));
        }
}
