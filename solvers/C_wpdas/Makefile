# CC = gcc
# CFLAGS = -g -Wall -O3
# LDFLAGS =
# PROG = wpdas
# LIBS = -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
# SRCS = wpdas_sg2.c main.c
# OBJS = $(patsubst %.c,%.o,$(SRCS))

# all: $(PROG)

# $(PROG) : $(OBJS)
# 	$(CC) $(LDFLAGS) $(CFLAGS) -o $(PROG) $(OBJS) $(LIBS)

# $(OBJS) : %.o : %.c
# 	$(CC) -c $(CFLAGS) $< -o $@

# .PHONY : clean 

# clean:
# 	rm -f *.o $(PROG)


CC = icc
CFLAGS = -mkl=parallel -O3
LDFLAGS =
PROG = wpdas
LIBS = -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
SRCS = wpdas_sg2.c main.c
OBJS = $(patsubst %.c,%.o,$(SRCS))

all: $(PROG)

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(PROG) $(OBJS) $(LIBS)

$(OBJS) : %.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean 

clean:
	rm -f *.o $(PROG)
