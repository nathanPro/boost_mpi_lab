#include "gpu.h"
#include "pic.h"
#include <cmath>
#include <ctime>
#include <omp.h>
#include <thrust/complex.h>

template <int M, typename REAL>
__device__ __host__ int iterate(thrust::complex<REAL> s) {
  thrust::complex<REAL> w = 0;
  int cnt = M;
  for (int i = 0; cnt == M && i < M; i++) {
    if (thrust::norm(w) > REAL(4.))
      cnt = i;
    w = w * w + s;
  }
  return cnt;
}

template <int M, typename REAL>
__global__ void calculate_indices(int width, int height,
                                  thrust::complex<REAL> c, REAL dx, REAL dy,
                                  png_byte *data) {
  using C = thrust::complex<REAL>;
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < width * height) {
    const int x = idx % width;
    const int y = idx / width;
    auto &pixel =
        reinterpret_cast<pic::canvas::rgba_reference &>(data[4 * idx]);
    pic::colorize(pixel, iterate<M>(c + C{x * dx, y * dy}));
  }
}

template <int M, typename REAL>
void gpu_calculate(int width, int height, thrust::complex<REAL> c[2],
                   int threads, const char *file) {
  using C = thrust::complex<REAL>;
  const REAL dx = (c[1] - c[0]).real() / REAL(width),
             dy = (c[1] - c[0]).imag() / REAL(height);

  gpu::array<png_byte> data(4 * width * height);
  int num_blocks = (width * height + threads - 1) / threads;
  calculate_indices<M>
      <<<num_blocks, threads>>>(width, height, c[0], dx, dy, data.data());

  png_byte *out;
  cudaMallocHost(&out, 4 * width * height);
  data.get(out);

  pic::canvas img(width, height, out);
  img.save(file);

  cudaFreeHost(out);
}
