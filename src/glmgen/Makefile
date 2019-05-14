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
OBJ=obj/*.o
IDIR=../include/
TARGET=lib/libglmgen${EXT}

# Compiler Dependent Environment Variables 
ifneq ($(CC),)
    CC = gcc
endif
CFLAGS=-O3 -Wall -Wextra -ansi -std=c89 -pedantic
CFLAGS2=-O3

# Recipes
all:
	mkdir -p lib
	mkdir -p obj
	cd obj; ${CC} ${CLFAGS2} -c -fPIC ../src/csparse/*.c -I${IDIR}
	cd obj; ${CC} ${CFLAGS}  -c -fPIC ../src/utils/*.c -I${IDIR}
	cd obj; ${CC} ${CFLAGS}  -c -fPIC ../src/tf/*.c -I${IDIR}
	${CC} ${LDFLAGS} -o ${TARGET} ${OBJ}

clean:
	rm -rf lib
	rm -rf obj