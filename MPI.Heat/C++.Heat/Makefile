USE_MPI := 
# -DUSE_MPI
EQ_FLAG := -DHEAT
# -DPOSSION
# -DHEAT
# Debug mode
USE_DEBUG := 
# -DDEBUG

MPICXX := mpic++

MPI_STD := -std=c++20

DFLAGS := 
# -W -Werror -Wextra
LFLAGS := 

MPI_LFLAGS := -lboost_mpi -lboost_serialization 
MPI_LPATH 	:= -L/opt/homebrew/Cellar/boost/1.85.0/lib \
							-L/opt/homebrew/Cellar/boost-mpi/1.85.0/lib 
MPI_IPATH := -I/opt/homebrew/Cellar/boost/1.85.0/include 

SRC_1 := main.cc

EXEC_1 := $(SRC_1:.cc=)

EXECS := $(EXEC_1)

all: $(EXECS)

$(EXECS): $(SRC_1)
	$(MPICXX) $(USE_DEBUG) $(EQ_FLAG) $(USE_MPI) $(MPI_STD) $(DFLAGS) $(MPI_IPATH) $(MPI_LPATH) $(MPI_LFLAGS) $^ -o $@ -O3


.PHONY: clean all
clean:
	$(RM) $(EXECS)
