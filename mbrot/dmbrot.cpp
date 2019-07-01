#include "fmt/core.h"
#include "fmt/printf.h"
#include "pic.h"
#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>

namespace mpi = boost::mpi;

extern void run_in_cpu(png_byte *data, std::complex<float> c0,
                       std::complex<float> c1, std::complex<float> delta,
                       size_t lo, size_t hi, size_t width, size_t height,
                       size_t threads);

extern void run_in_gpu(png_byte *data, std::complex<float> c0,
                       std::complex<float> c1, std::complex<float> delta,
                       size_t lo, size_t hi, size_t width, size_t height,
                       size_t threads);

int main(int argc, char **argv) {
    mpi::environment env;
    mpi::communicator world;

    const std::complex<float> c[2] = {{std::stof(argv[1]), std::stof(argv[2])},
                                      {std::stof(argv[3]), std::stof(argv[4])}};
    const size_t width = std::stoi(argv[5]);
    const size_t height = std::stoi(argv[6]);
    const bool cpu = (strcmp(argv[7], "cpu") == 0);
    const size_t threads = std::stoi(argv[8]);
    const std::string filename = argv[9];

    const size_t lines_per_proc = ceil(float(height) / world.size());
    const size_t lo = lines_per_proc * world.rank();
    const size_t hi = std::min(lo + lines_per_proc, height);
    const std::complex<float> delta = {real(c[1] - c[0]) / double(width),
                                       imag(c[1] - c[0]) / double(height)};

    const size_t bitmap_size
        = 4 * width * (world.rank() == 0 ? height : (hi - lo));
    std::vector<png_byte> map(bitmap_size);
    auto calculate = [&, dev = world.rank() % 2]() {
        if (cpu)
            run_in_cpu(map.data(), c[0], c[1], delta, lo, hi, width, height,
                       threads);
        else {
            cudaSetDevice(dev);
            run_in_gpu(map.data(), c[0], c[1], delta, lo, hi, width, height,
                       threads);
        }
    };

    if (world.rank() != 0) {
        calculate();
        world.send(0, 0, map.data(), map.size());
    } else {
        std::vector<mpi::request> transmissions(world.size() - 1);
        for (int i = 1, lo = lines_per_proc; i < world.size();
             i++, lo += lines_per_proc) {
            const size_t hi = std::min(lo + lines_per_proc, height);
            transmissions[i - 1] = world.irecv(
                i, 0, map.data() + 4 * width * lo, 4 * width * (hi - lo));
        }
        calculate();
        mpi::wait_all(begin(transmissions), end(transmissions));
        pic::canvas img(width, height, map.data());
        img.save(filename.c_str());
    }
}
