
gpu_obj.o:
	nvcc -x cu -I. -dc mbrot.cpp -o mbrot.o
	nvcc -x cu -I. -dc pic.cpp -o pic.o
	nvcc -dlink pic.o mbrot.o -o gpu_obj.o

dmbrot:	main.cpp gpu_obj.o
	mpic++ -c main.cpp -o dmbrot.o -I/usr/local/cuda-9.1/include
	mpic++ $(CUDA_FLAGS) -o dmbrot gpu_obj.o pic.o mbrot.o dmbrot.o -lpng -fopenmp

.PHONY: clean

clean:
	rm -f *.o dmbrot

test: dmbrot
	mpirun -n 4 ./dmbrot -1.92 -1.08 1.92 1.08 1920 1080 gpu 1024 gpu.png
	mpirun -n 4 ./dmbrot -1.92 -1.08 1.92 1.08 960 540 cpu 1024 cpu.png
	#mpirun -n 4 ./dmbrot -1.92 -1.08 1.92 1.08 17 13 cpu 1024 img.png
	#eog img.png
