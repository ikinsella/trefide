CC = gcc
CFLAGS = -g -Wall -O3
LDFLAGS =
PROG = pdas
LIBS = -lm -llapacke -llapack -lblas
SRCS = pdas_sg2.c main.c
OBJS = $(patsubst %.c,%.o,$(SRCS))

all: $(PROG)

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(PROG) $(OBJS) $(LIBS)

$(OBJS) : %.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean 

clean:
	rm -f *.o $(PROG)
