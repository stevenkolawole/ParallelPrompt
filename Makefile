# Variables
CC = g++ -std=c++20
CFLAGS = -Wall -Iinclude -Iinclude/openai
TARGET = alphabits
LDFLAGS = -lcurl
SRCDIR = src
OBJDIR = obj
BINDIR = bin
SOURCES := $(wildcard $(SRCDIR)/parallel_vary_n.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Rules
all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean

