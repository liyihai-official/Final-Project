ifdef USE_OMP
  OMP += -DUSE_OMP -fopenmp
endif

EQ_FLAG := -DHEAT # -DPOSSION # -DHEAT

# Debug mode 
USE_DEBUG := # -DDEBUG

MAX_N_X := -DMAX_N_X=12
MAX_N_Y := -DMAX_N_Y=12

MPICXX := mpic++

MPI_STD := -std=c++20

DFLAGS := #-W -Werror -Wextra
LFLAGS := 

# GL := -g -lprofiler


SRC_1 := main.cc
EXEC_1 := $(SRC_1:.cc=)

SRC_2 := 3d.cc
EXEC_2 := $(SRC_2:.cc=)

EXECS := $(EXEC_1) $(EXEC_2)

all: $(EXECS)

$(EXEC_1): $(SRC_1)
	$(MPICXX) $(USE_DEBUG) $(MAX_N_X) $(MAX_N_Y) $(EQ_FLAG) $(OMP) $(MPI_STD) $(DFLAGS) $(GL) $^ -o $@ -O2

$(EXEC_2): $(SRC_2)
	$(MPICXX) $(USE_DEBUG) $(MPI_STD) $(DFLAGS) $^ -o $@



.PHONY: clean all
clean:
	$(RM) $(EXECS)








MPI_LFLAGS := 	#-lboost_mpi -lboost_serialization 
MPI_LPATH 	:= 	#-L/opt/homebrew/Cellar/boost/1.85.0/lib \
								-L/opt/homebrew/Cellar/boost-mpi/1.85.0/lib 
MPI_IPATH :=  	#-I/opt/homebrew/Cellar/boost/1.85.0/include 