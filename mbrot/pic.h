#ifndef PIC_H
#define PIC_H

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <png.h>
#include <cuda_runtime.h>
#include <vector>

template <typename... Args>[[noreturn]] void hcf(Args... args) {
  fprintf(stderr, args...);
  abort();
}

namespace pic {

void error_png(png_structp, png_const_charp error_msg);
void warning_png(png_structp, png_const_charp msg);

struct file_guard {
  FILE *pointer;

  file_guard(const char *filename, const char *mode);
  ~file_guard();
};

class canvas {
public:
  struct rgba_reference {
    union {
      png_byte bytes[4];
      struct {
        uint8_t red, green, blue, alpha;
      };
    };
  };

  const int width;
  const int height;
  static const int bytes_per_pixel = 4;
  std::vector<png_byte *> rows;

private:
  std::vector<png_byte> data;

public:
  canvas(int w, int h) noexcept;
  canvas(int w, int h, png_byte *inner) noexcept;
  void save(const char *filename) noexcept;
  rgba_reference &operator()(int y, int x);
};

canvas::rgba_reference &linear_byte_access(int w, int, png_byte *data,
                                           int y, int x);

__device__ __host__ void colorize(canvas::rgba_reference &rgba,
                                  double cnt);
} // namespace pic

#endif
