CXX := g++-13

EQ_FLAG := -DHEAT
# -DPOSSION



all: main

main: main.cc
	$(CXX) $(EQ_FLAG) -std=c++20 -W -Werror -o $@ $^ 

.PHONY: clean

clean:
	$(RM) main
