CXX = icpc
CXXFLAGS = -mkl=sequential -qopenmp -O3 -fPIC -I$(shell pwd)/proxtv -I$(shell pwd)/glmgen/include/ -D NOMATLAB=1 #-mkl=parallel
LDFLAGS = -shared
PROXTV = proxtv
LIBPROXTV = $(PROXTV)/libproxtv.so
GLMGEN = glmgen
LIBGLMGEN = $(GLMGEN)/lib/libglmgen.so
LIBTREFIDE = libtrefide.so
LIBS = -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5
SRCS = utils/welch.cpp proxtf/wpdas.cpp proxtf/line_search.cpp proxtf/utils.cpp proxtf/l1tf/ipm.cpp proxtf/admm.cpp pmd/pmd.cpp pmd/decimation.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all
all: $(LIBPROXTV) $(LIBGLMGEN) $(LIBTREFIDE)

$(LIBPROXTV):
	cd $(PROXTV); make;

$(LIBGLMGEN):
	cd $(GLMGEN); make;

$(LIBTREFIDE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(LIBTREFIDE) $(OBJS) $(SRCS:.cpp=.d)
	cd $(PROXTV); make clean;
	cd $(GLMGEN); make clean;
