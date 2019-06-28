CXX=mpic++
BOOST_FLAGS=-Iboost -Lboost/stage/lib -lboost_mpi -lboost_serialization -Wl,-rpath,./boost/stage/lib
CXXFLAGS=--std=c++17 -Wall -Wextra $(BOOST_FLAGS)

local:
	mkdir local
	tar xvf boost_1_70_0_rc1.tar.gz
	mv boost_1_70_0 boost
	cd boost && ./bootstrap.sh --prefix=../local --with-libraries=mpi
	cd boost && echo "using mpi ;" >> project-config.jam
	cd boost && ./b2 -j4
