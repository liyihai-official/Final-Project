USE_MPI := # -DUSE_MPI

EQ_FLAG := -DHEAT # -DPOSSION # -DHEAT

# Debug mode 
USE_DEBUG := # -DDEBUG

MAX_N_X := -DMAX_N_X=32
MAX_N_Y := -DMAX_N_Y=32

MPICXX := mpic++

MPI_STD := -std=c++20

DFLAGS := #-W -Werror -Wextra
LFLAGS := 

MPI_LFLAGS := #-lboost_mpi -lboost_serialization 
MPI_LPATH 	:= 	#-L/opt/homebrew/Cellar/boost/1.85.0/lib \
								-L/opt/homebrew/Cellar/boost-mpi/1.85.0/lib 
MPI_IPATH :=  #-I/opt/homebrew/Cellar/boost/1.85.0/include 

GL := 		#-L/opt/homebrew/Cellar/gperftools/2.15/lib
GI := 		#-I/opt/homebrew/Cellar/gperftools/2.15/include

SRC_1 := main.cc
EXEC_1 := $(SRC_1:.cc=)
EXECS := $(EXEC_1) 


all: $(EXECS)

$(EXEC_1): $(SRC_1)
	$(MPICXX)$(USE_DEBUG) $(MAX_N_X) $(MAX_N_Y) $(EQ_FLAG) -fopenmp $(USE_MPI) $(MPI_STD) $(DFLAGS) $(MPI_IPATH) $(MPI_LPATH) $(MPI_LFLAGS) -g -lprofiler $^ -o $@ -O2

.PHONY: clean all
clean:
	$(RM) $(EXECS)
