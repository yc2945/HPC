
CXX = g++ # define a variable CXX
CXXFLAGS = -std=c++11 -O3 -march=native

ifeq "$(CXX)" "icpc" # conditionals
CXXFLAGS += -qopenmp # for Intel
else
CXXFLAGS += -fopenmp # for GCC
endif

TARGETS = $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< -o $@

%:%.c *.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS)

.PHONY: all, clean

