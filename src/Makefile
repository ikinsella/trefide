CXX = icpc
CXXFLAGS = -mkl=sequential -qopenmp -O3 -fPIC -I$(shell pwd)/proxtv -D NOMATLAB=1 #-mkl=parallel
LDFLAGS = -shared
DEP_DIR = proxtv
DEP_LIB = $(DEP_DIR)/libproxtv.so
TARGET_LIB = libtrefide.so
LIBS = -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5
SRCS = utils/welch.cpp proxtf/wpdas.cpp proxtf/line_search.cpp proxtf/utils.cpp proxtf/l1tf/ipm.cpp pmd/pmd.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all
all: $(DEP_LIB) $(TARGET_LIB)

$(DEP_LIB):
	cd $(DEP_DIR); make clean; make;

$(TARGET_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(TARGET_LIB) $(OBJS) $(SRCS:.cpp=.d)
