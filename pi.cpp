#include <algorithm>
#include <boost/format.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "fmt/core.h"
#include "fmt/printf.h"

using boost::format;
namespace mpi = boost::mpi;

int main(int argc, char **argv) {
    mpi::environment env;
    mpi::communicator world;

    if (argc == 1) {
        if (!world.rank()) fmt::printf("USAGE:%s\tn\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t n = atoi(argv[1]);
    const size_t points_per_process = (n + world.size() - 1u) / world.size();
    const size_t lo = world.rank() * points_per_process;
    const size_t hi = std::min(lo + points_per_process, n);
    const double dx = 1. / double(n);

    double I = 0.;
    {
        double x = (lo + .5) * dx;
        for (size_t i = lo; i < hi; i++, x += dx) {
            I += sqrt(1. - x * x) * dx;
        }
        if (world.rank() != 0) world.send(0, 0, I);
    }

    if (world.rank() == 0) {
        for (int i = 1; i < world.size(); i++) {
            double msg;
            world.recv(mpi::any_source, 0, msg);
            I += msg;
        }
        std::cout << format("%.12f\n") % (4. * I);
    }
}
