CLANG_BIN := /home/depctg/EP2-MLIR/llvm-project/build/bin
CC := $(CLANG_BIN)/clang

LDFLAGS = $(shell pkgconf -libs glib-2.0) -lpthread -target $(TARGET)
CFLAGS = -O2 $(shell pkgconf --cflags glib-2.0) -gdwarf-4 -target $(TARGET)

OBJECTS = runtime.o driver.o externs.o ep2.o ep2.inc.o
BIN = runep2

all : $(BIN)

runep2 : $(OBJECTS)
	$(CC) -gdwarf-4 -o $@ $^ $(LDFLAGS)

# already generated
ep2.o : ;

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f *.o *.inc.* $(BIN)
