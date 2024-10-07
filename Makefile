CC=gcc
CFLAGS=-O3

LD=gcc
LD_FLAGS=-lm

BUILD_DIR=build
BUILD_OBJ_DIR=$(BUILD_DIR)/obj
BUILD_OUTPUT=$(BUILD_DIR)/output

run: build
	$(BUILD_OUTPUT)

build: build-obj
	$(LD) $(BUILD_OBJ_DIR)/*.o $(LD_FLAGS) -o $(BUILD_OUTPUT)

build-obj: setup main neuron mathf train layer

%: %.c
	$(CC) $(CFLAGS) -c $^ -o $(BUILD_OBJ_DIR)/$@.o

setup:
	mkdir -p $(BUILD_OBJ_DIR)
