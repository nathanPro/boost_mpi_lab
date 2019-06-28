#include "pic.h"

namespace pic {

void error_png(png_structp, png_const_charp error_msg) {
  hcf(error_msg);
}

void warning_png(png_structp, png_const_charp msg) {
  fprintf(stderr, msg);
}

file_guard::file_guard(const char *filename, const char *mode)
    : pointer(fopen(filename, mode)) {
  if (!pointer)
    hcf("Could not open file\n");
}

file_guard::~file_guard() { fclose(pointer); }

canvas::rgba_reference &linear_byte_access(int w, int, png_byte *data,
                                           int y, int x) {
  return reinterpret_cast<canvas::rgba_reference &>(
      data[y * 4 * w + 4 * x]);
}

canvas::canvas(int w, int h) noexcept
    : width(w), height(h), rows(height),
      data(width * height * bytes_per_pixel) {
  for (int y = 0; y < h; y++)
    rows[y] = data.data() + y * (width * bytes_per_pixel);
}

canvas::canvas(int w, int h, png_byte *inner) noexcept
    : width(w), height(h), rows(height),
      data(inner, inner + 4 * width * height) {
  for (int y = 0; y < h; y++)
    rows[y] = data.data() + y * (width * bytes_per_pixel);
}

void canvas::save(const char *filename) noexcept {
  png_structp png_ptr;
  png_infop info_ptr;
  file_guard file(filename, "wb");

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,
                                    error_png, warning_png);

  if (!png_ptr)
    hcf("png_create_write_struct error\n");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    hcf("png_create_info_strucvt error\n");

  png_init_io(png_ptr, file.pointer);
  png_set_IHDR(png_ptr, info_ptr, width, height, 8,
               PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_write_image(png_ptr, rows.data());
  png_write_end(png_ptr, info_ptr);
}

canvas::rgba_reference &canvas::operator()(int y, int x) {
  return linear_byte_access(width, height, data.data(), y, x);
}

__device__ __host__ void colorize(canvas::rgba_reference &rgba,
                                  double cnt) {
  uint32_t out = 0xB2DDF7;
  double coef = exp(-0.05 * cnt);

  // clang-format off
  rgba.blue  = floor(coef * (out & 0xFF)); out >>= 8;
  rgba.green = floor(coef * (out & 0xFF)); out >>= 8;
  rgba.red   = floor(coef * (out & 0xFF));
  rgba.alpha = 255;
  // clang-format on
}
} // namespace pic
