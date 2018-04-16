CC = icc
CFLAGS = -mkl=parallel -O3
LDFLAGS =
PDAS = wpdas
IPM = ipm
LS = cpdas
LIBS = -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
#LIBS = -lblas -llapack -lm
PDAS_SRCS = utils.c wpdas.c wpdas_main.c
IPM_SRCS = utils.c ipm.c ipm_main.c
LS_SRCS = utils.c wpdas.c line_search.c ls_main.c
PDAS_OBJS = $(patsubst %.c,%.o,$(PDAS_SRCS))
IPM_OBJS = $(patsubst %.c,%.o,$(IPM_SRCS))
LS_OBJS = $(patsubst %.c,%.o,$(LS_SRCS))

all: $(PDAS) $(IPM) $(LS)

$(PDAS) : $(PDAS_OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(PDAS) $(PDAS_OBJS) $(LIBS)

$(IPM) : $(IPM_OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(IPM) $(IPM_OBJS) $(LIBS)

$(LS) : $(LS_OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(LS) $(LS_OBJS) $(LIBS)

$(PDAS_OBJS) : %.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

$(IPM_OBJS) : %.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean 

clean:
	rm -f *.o $(PDAS) $(IPM) $(LS)
