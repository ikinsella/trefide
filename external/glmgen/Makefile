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

# If undefin. not using conda-build), set PREFIX to active env
PREFIX ?= $(CONDA_PREFIX)

# Project Structure Dependent Variables
IDIR = include
TARGET = $(PREFIX)/lib/libglmgen${EXT}

# Compiler Dependent Environment Variables
CC ?= gcc
#ifneq ($(CC),)
#    CC = gcc
#endif

INCLUDES = -I$(IDIR)
CFLAGS = -O3 -Wall -Wextra -ansi -std=c89 -pedantic -fPIC $(INCLUDES)

SRCS = $(wildcard src/csparse/*.c)
SRCS += $(wildcard src/utils/*.c)
SRCS += $(wildcard src/tf/*.c)
OBJS = $(patsubst %.c,%.o,$(SRCS))

all: clean $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)
