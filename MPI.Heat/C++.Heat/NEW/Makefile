


MAX_N_X := -DMAX_N_X=12
MAX_N_Y := -DMAX_N_Y=12

MPICXX := mpic++
MPI_STD := -std=c++20

DFLAGS := #-W -Werror -Wextra
LFLAGS := 


SRC_1 := main.cc
EXEC_1 := $(SRC_1:.cc=)


EXECS := $(EXEC_1)

all: $(EXECS)

$(EXEC_1): $(SRC_1)
	$(MPICXX) $(MAX_N_X) $(MAX_N_Y) $(MPI_STD) $(DFLAGS) $^ -o $@ -O2


.PHONY: clean all
clean:
	$(RM) $(EXECS)