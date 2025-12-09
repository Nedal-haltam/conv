#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "PCIE.h"

#define DEMO_PCIE_USER_BAR			PCIE_BAR4
#define DEMO_PCIE_IO_LED_ADDR		0x4000010
#define DEMO_PCIE_IO_BUTTON_ADDR	0x4000020
#define DEMO_PCIE_MEM_ADDR			0x00000000
#define MEM_SIZE 					(512*1024)

void *pcie_lib_handle;
PCIE_HANDLE hPCIE;
const PCIE_LOCAL_ADDRESS start_address = DEMO_PCIE_MEM_ADDR;

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

void test_pcie()
{
	PCIe_Prologue();
	const bool check = 1;
	const bool print = 0;
	const int n = MEM_SIZE;
	uint8_t writebuff[n] = {0};
	uint8_t readbuff[n] = {0};

	for (int i = 0; i < n; i++) *(writebuff + i) = i + 1;

	PCIeWrite(writebuff, 0, n * sizeof(writebuff[0]));
	PCIeRead(readbuff, 0, n * sizeof(readbuff[0]));

	for (int i = 0; i < n; i++) {
		if (print) printf("written=%d read=%d\n", *(writebuff + i), *(readbuff + i));
		if (check && *(writebuff + i) != *(readbuff + i))
		{
			printf("written=%d read=%d\n", *(writebuff + i), *(readbuff + i));
			printf("address(bytes): %d\n", i);
			printf("not matched\n");
			break;
		}
	}

	PCIe_Epilogue();
}

int main(int argc, char* argv[])
{
	test_pcie();
	return 0;
}

