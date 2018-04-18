CC = icc
CFLAGS = -mkl=sequential -O3 -fPIC #-mkl=parallel -qopenmp
CXX = icpc
CXXFLAGS = -mkl=sequential -O3 -fPIC #-mkl=parallel -qopenmp
LDFLAGS = -shared
TARGET_LIB = libtrefide.so
LIBS = -lmkl_intel_lp64 -lmkl_core -lm  #-lmkl_intel_thread -liomp5
SRCS = welch.cpp wpdas.cpp line_search.cpp utils.cpp ipm.cpp pmd.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all
all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CC) $(CFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(TARGET_LIB) $(OBJS) $(SRCS:.cpp=.d)
