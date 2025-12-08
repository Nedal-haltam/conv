.PHONY: all build run

all: build run

build:
	g++ main.cpp ./deps/PCIE.c -I./deps -o main -L. -l:./deps/terasic_pcie_qsys.so -ldl -lpthread -O3 -march=native
# -lssl -lcrypto -lpcap 
run:
	sudo ./main
