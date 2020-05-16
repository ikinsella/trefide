# Detect Operating System
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

# TODO: Set Windows Specific Environment Variables
ifeq ($(detected_OS),Windows)
    echo "Installation on Windows not currently supported."
endif

# Set MacOS Specific Environment Variables
ifeq ($(detected_OS),Darwin)
    EXT=.dylib
    LDFLAGS = -dynamiclib
endif

# Set Linux Specific Environment Variables
ifeq ($(detected_OS),Linux)
    EXT=.so
    LDFLAGS = -shared
endif

# Project Structure Dependent Variables
PROXTV = $(shell pwd)/external/proxtv
LIBPROXTV = $(PROXTV)/libproxtv$(EXT)

GLMGEN = $(shell pwd)/external/glmgen
LIBGLMGEN = $(GLMGEN)/lib/libglmgen$(EXT)

LIBTREFIDE = libtrefide$(EXT)

LDLIBS = -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5

SRCS = src/welch.cpp src/wpdas.cpp src/line_search.cpp src/utils.cpp src/ipm.cpp src/admm.cpp src/pmd.cpp src/decimation.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

# LDFLAGS += -L$(PROXTV) -L$(GLMGEN)/lib
INCLUDES = -I$(GLMGEN)/include -I$(PROXTV)

WARNINGS := -Wall -Wextra -pedantic -Weffc++ -Wshadow -Wpointer-arith \
            -Wcast-align -Wwrite-strings -Wmissing-declarations \
            -Wredundant-decls -Winline -Wno-long-long -Wconversion \
            -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wrestrict \
            -Wnull-dereference -Wold-style-cast -Wuseless-cast \
            -Wdouble-promotion -Wformat=2

CXXFLAGS := $(WARNINGS) -O3

# Compiler Dependent Environment Variables
ifeq ($(CXX),)
    CXX = g++
endif
ifeq ($(CXX), icpc)
    CXXFLAGS += -mkl=sequential -qopenmp -fPIC $(INCLUDES) $(LDFLAGS) -D NOMATLAB=1
else
    CXXFLAGS += -fopenmp -fPIC $(INCLUDES) $(LDFLAGS) -D NOMATLAB=1
endif

# Recipes
.PHONY: all
all: clean $(LIBTREFIDE) $(LIBGLMGEN) $(LIBPROXTV)

$(LIBTREFIDE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

#$(SRCS:.cpp=.d):%.d:%.cpp
#	$(CXX) $(CXXFLAGS) -o $@ $^

$(LIBPROXTV):
	$(MAKE) -C $(PROXTV);

$(LIBGLMGEN):
	$(MAKE) -C $(GLMGEN);

.PHONY: clean
clean:
	rm -f $(LIBTREFIDE) $(OBJS)
	$(MAKE) clean -C $(PROXTV);
	$(MAKE) clean -C $(GLMGEN);
