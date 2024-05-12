CC = gcc-13
EXES := heat3dSeq
OBJS := heatUtilSeq heat3dSeq
OPTIM := -O3
DFLAGS := -Wall -Wextra

SRCS := $(EXES:=.c)

all: $(EXES)

%.o: %.c
	$(CC) -c $< -o $@ -lm
$(EXES): $(OBJS:=.o)
	$(CC) -o $@ $^

.PHONY: clean

clean:
	$(RM) $(EXES) $(OBJS:=.o)
