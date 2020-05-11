# Detect Operating System
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

# Set Windows Specific Environment Variables: TODO
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
PROXTV = $(shell pwd)/proxtv
LIBPROXTV = $(PROXTV)/libproxtv$(EXT)

GLMGEN = $(shell pwd)/glmgen
LIBGLMGEN = $(GLMGEN)/lib/libglmgen$(EXT)

LIBTREFIDE = libtrefide$(EXT)

LDLIBS = -lproxtv -lglmgen -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5
SRCS = utils/welch.cpp proxtf/wpdas.cpp proxtf/line_search.cpp proxtf/utils.cpp proxtf/l1tf/ipm.cpp proxtf/admm.cpp pmd/pmd.cpp pmd/decimation.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

LDFLAGS += -L$(PROXTV) -L$(GLMGEN)/lib
INCLUDES = -I$(PROXTV) -I$(GLMGEN)/include

CXXFLAGS = -Wall -Wextra -Weffc++ -pedantic -O3

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
all: clean $(LIBTREFIDE)

$(LIBTREFIDE): $(OBJS) $(LIBPROXTV) $(LIBGLMGEN)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(SRCS:.cpp=.d):%.d:%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

$(LIBPROXTV):
	$(MAKE) -C $(PROXTV);

$(LIBGLMGEN):
	$(MAKE) -C $(GLMGEN);

.PHONY: clean
clean:
	rm -f $(LIBTREFIDE) $(OBJS) $(SRCS:.cpp=.d)
	$(MAKE) clean -C $(PROXTV);
	$(MAKE) clean -C $(GLMGEN);
