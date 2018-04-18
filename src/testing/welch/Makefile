CC = icc
CFLAGS = -mkl=parallel -O3
LDFLAGS =
EXEC = welch
LIBS = -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
SRCS = welch.c test_welch.c
OBJS = $(patsubst %.c,%.o,$(SRCS))

all: $(EXEC)

$(EXEC) : $(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

$(OBJS) : %.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean 

clean:
	rm -f *.o $(EXEC)
