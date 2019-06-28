BOOST_FLAGS=-Iboost -Lboost/stage/lib -lboost_mpi -lboost_serialization -Wl,-rpath,./boost/stage/lib
CXXFLAGS=--std=c++17 -Wall -Wextra -Ifmt/include

pi: bin boost fmt.o pi.cpp
	mpic++ $(CXXFLAGS) $(BOOST_FLAGS) bin/fmt.o pi.cpp -o bin/pi

fmt.o: bin fmt/src/format.cc
	g++ $(CXXFLAGS) -c fmt/src/format.cc -o bin/fmt.o

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
