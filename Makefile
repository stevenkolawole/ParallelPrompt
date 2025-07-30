# Variables
CC = g++ -std=c++20
CFLAGS = -Wall -Iinclude -Iinclude/openai
SRC = src/serial_vs_parallel.cpp
OUT = bin/alphabits
LDFLAGS = -lcurl

# Default target
all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p $(dir $(OUT))
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(OUT)

.PHONY: all clean
