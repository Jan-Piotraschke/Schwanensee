SHELL := /bin/bash

# Paths
PROJECT_DIR := $(shell pwd)
ARCHITECTURE := $(shell uname -m)

# Platform selection
BUILD_DIR = build
CMAKE_CMD = cmake
BUILD_CMD = ninja

# Determine OS
OS := $(shell uname -s)

# Compiler paths
ifeq ($(OS), Darwin)
	C_COMPILER=$(shell brew --prefix llvm)/bin/clang
	CXX_COMPILER=$(shell brew --prefix llvm)/bin/clang++
else ifeq ($(OS), Linux)
	C_COMPILER=/usr/bin/gcc
	CXX_COMPILER=/usr/bin/g++
endif

.PHONY: all
all: build_cpp

.PHONY: build_cpp
build_cpp:
	@echo "Building for platform: $(PLATFORM)"; \
	$(CMAKE_CMD) -S $(PROJECT_DIR) \
			-B $(PROJECT_DIR)/$(BUILD_DIR) \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_C_COMPILER=$(C_COMPILER) \
			-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
			-DCMAKE_PREFIX_PATH=$(PROJECT_DIR)/libtorch \
			-DCMAKE_CXX_STANDARD=17 \
			-DCMAKE_OSX_ARCHITECTURES=$(ARCHITECTURE) \
			-GNinja
ifeq ($(OS), Darwin)
	$(BUILD_CMD) -C $(PROJECT_DIR)/$(BUILD_DIR) -j $(shell sysctl -n hw.logicalcpu)
else ifeq ($(OS), Linux)
	$(BUILD_CMD) -C $(PROJECT_DIR)/$(BUILD_DIR) -j $(shell nproc)
endif
