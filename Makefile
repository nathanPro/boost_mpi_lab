BOOST_FLAGS=-Iboost -Lboost/stage/lib -lboost_mpi -lboost_serialization -Wl,-rpath,./boost/stage/lib
CUDA_FLAGS=-L/usr/local/cuda-9.1/lib64/ -lcuda -lcudart 
CXXFLAGS=--std=c++17 -Wall -Wextra -Ifmt/include

pi: bin boost fmt.o pi.cpp
	mpic++ $(CXXFLAGS) $(BOOST_FLAGS) bin/fmt.o pi.cpp -o bin/pi

fmt.o: bin fmt/src/format.cc
	g++ $(CXXFLAGS) -c fmt/src/format.cc -o bin/fmt.o

bin/dmbrot: bin mbrot/dmbrot.cpp
	$(MAKE) -C mbrot 
	mpic++ $(CXXFLAGS) $(BOOST_FLAGS) -c mbrot/dmbrot.cpp -o mbrot/dmbrot.o -I/usr/local/cuda-9.1/include 
	mpic++ $(CUDA_FLAGS) $(BOOST_FLAGS) -o bin/dmbrot mbrot/gpu_obj.o mbrot/pic.o mbrot/mbrot.o mbrot/dmbrot.o -lpng -fopenmp

boost:
	tar xvf boost_1_70_0_rc1.tar.gz
	mv boost_1_70_0 boost
	cd boost && ./bootstrap.sh --with-libraries=mpi
	cd boost && echo "using mpi ;" >> project-config.jam
	cd boost && ./b2 -j4

bin:
	mkdir bin

clean:
	rm -rf bin
