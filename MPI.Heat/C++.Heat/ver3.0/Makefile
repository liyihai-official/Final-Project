
MAX_N_X := -DMAX_N_X=52
MAX_N_Y := -DMAX_N_Y=52
MAX_N_Z := -DMAX_N_Z=52

MPICXX := mpic++
MPI_STD := -std=c++20

DFLAGS := 
# -W -Werror -Wextra
LFLAGS := 


SRC_1 := main_1d.cc
EXEC_1 := $(SRC_1:.cc=)

SRC_2 := main_2d.cc
EXEC_2 := $(SRC_2:.cc=)

SRC_3 := main_3d.cc
EXEC_3 := $(SRC_3:.cc=)


EXECS := $(EXEC_1) $(EXEC_2) $(EXEC_3)

all: $(EXECS)

$(EXEC_1): $(SRC_1)
	$(MPICXX) $(MAX_N_X) $(MPI_STD) $(DFLAGS) $^ -o $@ -O2

$(EXEC_2): $(SRC_2)
	$(MPICXX) $(MAX_N_X) $(MAX_N_Y) $(MPI_STD) $(DFLAGS) $^ -o $@ -O2

$(EXEC_3): $(SRC_3)
	$(MPICXX) $(MAX_N_X) $(MAX_N_Y) $(MAX_N_Z) $(MPI_STD) $(DFLAGS) $^ -o $@ -O2


.PHONY: clean all
clean:
	$(RM) $(EXECS)