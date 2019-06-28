#include "pic.h"
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <string>

extern void run_in_cpu(png_byte *data, std::complex<float> c0,
                       std::complex<float> c1, std::complex<float> delta,
                       size_t lo, size_t hi, size_t width, size_t height,
                       size_t threads, size_t blocks, size_t M);

extern void run_in_gpu(png_byte *data, std::complex<float> c0,
                       std::complex<float> c1, std::complex<float> delta,
                       size_t lo, size_t hi, size_t width, size_t height,
                       size_t threads, size_t blocks, size_t M);

void run_dummy(bool cpu, png_byte *map, std::complex<float> c0,
               std::complex<float> c1, std::complex<float> delta, size_t lo,
               size_t hi, size_t W, size_t H, size_t threads, size_t blocks,
               size_t M) {
  if (cpu)
    run_in_cpu(map, c0, c1, delta, lo, hi, W, H, threads, blocks, M);
  else
    run_in_gpu(map, c0, c1, delta, lo, hi, W, H, threads, blocks, M);
}

int main(int argc, char **argv) {
  // parsear os argumentos {
  const std::complex<float> c0 = {std::stof(argv[1]), std::stof(argv[2])};
  const std::complex<float> c1 = {std::stof(argv[3]), std::stof(argv[4])};
  const size_t W = std::stoi(argv[5]);
  const size_t H = std::stoi(argv[6]);
  const bool cpu = (strcmp(argv[7], "cpu") == 0);
  const size_t threads = std::stoi(argv[8]);
  const std::string filename = argv[9];
  size_t blocks = 0; // 0 pra escolher automaticamente o número de blocos
  size_t M;

  if (argc > 10) {
    M = std::stof(argv[10]);

    if (argc > 11)
      blocks = std::stoi(argv[11]);
  } else
    M = 200;
  // parsear os argumentos }

  const std::complex<float> delta = {(real(c1) - real(c0)) / W,
                                     (imag(c1) - imag(c0)) / H};

  // número de processos
  // posição desse carinha na ordem de processos
  // tamanho do nome do processador nome do processador
  
  int num_proc, rank, proc_name_len;
  char proc_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(proc_name, &proc_name_len);

  // calcular a minha parte
  const size_t lines_per_proc = ceil((float)H / num_proc);
  const size_t lo = lines_per_proc * rank;
  const size_t hi = std::min(lo + lines_per_proc, H);

  // Host aloca matriz inteira, demais apenas sua parte
  const size_t bitmap_size = 4 * W * (rank == 0 ? H : (hi - lo));
  std::vector<png_byte> map(bitmap_size);
  run_dummy(cpu, map.data(), c0, c1, delta, lo, hi, W, H, threads, blocks, M);

  if (rank != 0) {
    MPI_Send(map.data(), (hi - lo) * W, MPI_UNSIGNED,
             0, // destino = host
             0, MPI_COMM_WORLD);
  } else {
    for (size_t i = 1, lo = lines_per_proc; i < num_proc;
         i++, lo += lines_per_proc) {
      const size_t hi = std::min(lo + lines_per_proc, H);
      MPI_Recv(map.data() + (4 * lo * W), (hi - lo) * W, MPI_UNSIGNED,
               i, // source
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    pic::canvas img(W, H, map.data());
    img.save(filename.c_str());
  }

  MPI_Finalize();
  return 0;
}
