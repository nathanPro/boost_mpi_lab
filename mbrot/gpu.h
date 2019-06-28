#include <vector>
#include <cstdlib>
#ifndef GPU_H
#define GPU_H

namespace gpu {
template <typename T> struct array {
  T *__beg;
  size_t __bytes;

public:
  array(size_t n) : __bytes(sizeof(T) * n) {
    cudaMalloc(&__beg, __bytes);
  }

  array(const std::vector<T> &src) {
    __bytes = src.size() * sizeof(T);
    cudaMalloc(&__beg, __bytes);
    cudaMemcpy(__beg, src.data(), __bytes, cudaMemcpyHostToDevice);
  }

  std::vector<T> get() const {
    std::vector<T> ans(size());
    get(ans.data());
    return ans;
  }

  void get(void* out) const {
    cudaMemcpy(out, __beg, __bytes, cudaMemcpyDeviceToHost);
  }

  const T *data() const { return __beg; }
  T *data() { return __beg; }
  size_t size() const { return __bytes / sizeof(T); }

  ~array() { cudaFree(__beg); }
};
} // namespace gpu

#endif
