CC = gcc-13
EXES := heat2dSeq
OBJS := heatUtilSeq heat2dSeq
SRCS := $(EXES:=.c)

all: $(EXES)

%.o: %.c
	$(CC) -c $< -o $@

$(EXES): $(OBJS:=.o)
	$(CC) -o $@ $^

.PHONY: clean

clean:
	$(RM) $(EXES) $(OBJS:=.o)
