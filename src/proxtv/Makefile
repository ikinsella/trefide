CXX = icpc
CXXFLAGS = -mkl=sequential -O3 -fPIC -D NOMATLAB=1 #-qopenmp
LDFLAGS = -shared
TARGET_LIB = libproxtv.so
LIBS = -lmkl_intel_lp64 -lmkl_core -lm  # -lmkl_intel_thread -liomp5
SRCS = condat_fast_tv.cpp johnsonRyanTV.cpp TVL1opt_kolmogorov.cpp LPopt.cpp TV2Dopt.cpp TV2DWopt.cpp TVgenopt.cpp TVL1opt.cpp TVL1opt_tautstring.cpp TVL1opt_hybridtautstring.cpp TVL1Wopt.cpp TVL2opt.cpp TVLPopt.cpp TVNDopt.cpp utils.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all
all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(TARGET_LIB) $(PDAS) $(OBJS) $(SRCS:.cpp=.d)
