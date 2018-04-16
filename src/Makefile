CC = icc
CFLAGS = -mkl=parallel -qopenmp -O3
LDFLAGS = -shared
TARGET_LIB = libtrefide.so
LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lm
SRCS = noise/welch.c tf/wpdas.c tf/line_search.c tf/utils.c tf/ipm.c
OBJS = $(patsubst %.c,%.o,$(SRCS))

.PHONY: all
all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.c=.d) : %.d : %.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(TARGET_LIB) $(OBJS) $(SRCS:.c=.d)
