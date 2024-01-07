CLANG_BIN := /home/depctg/EP2-MLIR/llvm-project/build/bin
CC := $(CLANG_BIN)/clang

LDFLAGS = $(shell pkgconf -libs glib-2.0) -lpthread
CFLAGS = -O2 $(shell pkgconf --cflags glib-2.0) -gdwarf-4

OBJECTS = runtime.o main.o externs.o gen.o
BIN = runep2

runep2 : $(OBJECTS)
	$(CC) -gdwarf-4 -o $@ $^ $(LDFLAGS)

gen.o : gen.ll
	$(CC) -gdwarf-4 -c -o $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f $(OBJECTS) $(BIN)
