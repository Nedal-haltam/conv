#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <inttypes.h>
#include "PCIE.h"

#define DEMO_PCIE_USER_BAR			PCIE_BAR4
#define DEMO_PCIE_IO_LED_ADDR		0x4000010
#define DEMO_PCIE_IO_BUTTON_ADDR	0x4000020
#define DEMO_PCIE_MEM_ADDR			0x00000000
#define MEM_SIZE 					(512*1024)

void *pcie_lib_handle;
PCIE_HANDLE hPCIE;
typedef __uint128_t word;

void PCIe_Prologue()
{
	pcie_lib_handle = PCIE_Load();
	if (!pcie_lib_handle) {
		printf("PCIE_Load failed!\n");
		exit(1);
	}
	hPCIE = PCIE_Open(DEFAULT_PCIE_VID, DEFAULT_PCIE_DID, 0);
	if (!hPCIE) {
		printf("PCIE_Open failed\n");
		exit(1);
	}
}

void PCIe_Epilogue()
{
	PCIE_Close(hPCIE);
	PCIE_Unload(pcie_lib_handle);
}

void PCIeWrite(void* data, PCIE_LOCAL_ADDRESS address, uint32_t len)
{
	if (!PCIE_DmaWrite(hPCIE, address, data, len))
	{
		printf("DMA Memory:PCIE_DmaWrite failed\n");
		PCIe_Epilogue();
		exit(1);
	}
}
void PCIeRead(void* data, PCIE_LOCAL_ADDRESS address, uint32_t len)
{
	if (!PCIE_DmaRead(hPCIE, address, data, len)) {
		printf("DMA Memory:PCIE_DmaRead failed\n");
		PCIe_Epilogue();
		exit(1);
	}
}

void pcie_rdwr_test()
{
	PCIe_Prologue();

	const bool check = 1;
	const bool print = 0;
	const int n = MEM_SIZE;
	uint8_t writebuff[n] = {0};
	uint8_t readbuff[n] = {0};

	for (int i = 0; i < n; i++) writebuff[i] = i + 1;

	PCIeWrite(writebuff, 0, n * sizeof(writebuff[0]));
	PCIeRead(readbuff, 0, n * sizeof(readbuff[0]));

	for (int i = 0; i < n; i++) {
		if (print) printf("written=%d read=%d\n", writebuff[i], readbuff[i]);
		if (check && writebuff[i] != readbuff[i])
		{
			printf("written=%d read=%d\n", writebuff[i], readbuff[i]);
			printf("address(bytes): %d\n", i);
			printf("not matched\n");
			break;
		}
	}

	PCIe_Epilogue();
}

void pcie_byte_test()
{
	PCIe_Prologue();

	uint64_t writebuff = 0;
	uint64_t readbuff = 0;
	writebuff = 0xFFFFFFFFFFFFFFFF;
	PCIeWrite(&writebuff, 0, 8);
	PCIeRead(&readbuff, 4, 8);

	printf("written=0x%"PRIx64"\n", writebuff);
	printf("read=0x%"PRIx64"\n", readbuff);

	PCIe_Epilogue();
}
#define WORD_MEM_SIZE (32768)
#define MMIO_ALU_in1   (16 * (WORD_MEM_SIZE - 1))
#define MMIO_ALU_in2   (16 * (WORD_MEM_SIZE - 2))
#define MMIO_ALU_out   (16 * (WORD_MEM_SIZE - 3))
#define MMIO_ALU_start (16 * (WORD_MEM_SIZE - 4))
#define MMIO_ALU_done  (16 * (WORD_MEM_SIZE - 5))

#include <unistd.h>

int main(int argc, char* argv[])
{
	// pcie_rdwr_test();
	// pcie_byte_test();

	PCIe_Prologue();
	word one = 1;
	word alu_in1 = 10;
	word alu_in2 = 20;
	word alu_out = 0;
	word alu_start = 123;

	PCIeWrite(&alu_in1, MMIO_ALU_in1, sizeof(alu_in1));
	PCIeWrite(&alu_in2, MMIO_ALU_in2, sizeof(alu_in2));
	PCIeWrite(&one, MMIO_ALU_start, sizeof(one));

	// word alu_done = 0;
	// while(!alu_done)
	// {
	// 	PCIeRead(&alu_done, MMIO_ALU_done, sizeof(alu_done));
	// }

	std::cout << "mm alu is DONE!\n";
	PCIeRead(&alu_in1, MMIO_ALU_in1, sizeof(alu_in1));
	PCIeRead(&alu_in2, MMIO_ALU_in2, sizeof(alu_in2));
	PCIeRead(&alu_out, MMIO_ALU_out, sizeof(alu_out));
	PCIeRead(&alu_start, MMIO_ALU_start, sizeof(alu_start));

	printf("alu_in1 = %llu\n", alu_in1);
	printf("alu_in2 = %llu\n", alu_in2);
	printf("alu_out = %llu\n", alu_out);
	printf("alu_start = %llu\n", alu_start);

	PCIe_Epilogue();

	return 0;
}

