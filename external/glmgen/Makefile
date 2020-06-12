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

# Project Structure Dependent Variables
# OBJ = obj/*.o
IDIR = include
TARGET = lib/libglmgen${EXT}

# Compiler Dependent Environment Variables
ifneq ($(CC),)
    CC = gcc
endif

INCLUDES = -I$(IDIR)
CFLAGS = -O3 -Wall -Wextra -ansi -std=c89 -pedantic -fPIC $(INCLUDES)

SRCS = $(wildcard src/csparse/*.c)
SRCS += $(wildcard src/utils/*.c)
SRCS += $(wildcard src/tf/*.c)
OBJS = $(patsubst %.c,%.o,$(SRCS))

all: clean $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET); rm -rf lib

$(TARGET): $(OBJS)
	mkdir -p lib
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)
