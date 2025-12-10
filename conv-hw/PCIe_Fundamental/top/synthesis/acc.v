`define BIT_WIDTH [127:0]
`define MEM_SIZE (32768)

`define CPU_AddressBus ((mem_write_en) ? mm_interconnect_0_onchip_memory2_s1_address	  : mm_interconnect_1_onchip_memory2_s2_address)
// `define MMIO_rden (mem_read_en  && (((`MEM_SIZE - 1024) <= `CPU_AddressBus) && (`CPU_AddressBus <= (`MEM_SIZE - 1))))
`define MMIO_rden (mem_read_en  && (((1) <= `CPU_AddressBus) && (`CPU_AddressBus <= (1024))))
// `define MMIO_wren (mem_write_en && (((`MEM_SIZE - 1024) <= `CPU_AddressBus) && (`CPU_AddressBus <= (`MEM_SIZE - 1))))
`define MMIO_wren (mem_write_en && (((1) <= `CPU_AddressBus) && (`CPU_AddressBus <= (1024))))

`define MMIO_ALU_in1_rden   (mem_read_en  && (`CPU_AddressBus == (1)))
`define MMIO_ALU_in1_wren   (mem_write_en && (`CPU_AddressBus == (1)))
`define MMIO_ALU_in2_rden   (mem_read_en  && (`CPU_AddressBus == (2)))
`define MMIO_ALU_in2_wren   (mem_write_en && (`CPU_AddressBus == (2)))
`define MMIO_ALU_out_rden   (mem_read_en  && (`CPU_AddressBus == (3)))
`define MMIO_ALU_out_wren   (mem_write_en && (`CPU_AddressBus == (3)))
`define MMIO_ALU_start_rden (mem_read_en  && (`CPU_AddressBus == (4)))
`define MMIO_ALU_start_wren (mem_write_en && (`CPU_AddressBus == (4)))
`define MMIO_ALU_done_rden  (mem_read_en  && (`CPU_AddressBus == (5)))
`define MMIO_ALU_done_wren  (mem_write_en && (`CPU_AddressBus == (5)))


wire clk;
wire `BIT_WIDTH CPUDataBusOut;

// MMIO
reg `BIT_WIDTH MMIODataBus;
//------------------------------------------------
// mm_alu
reg `BIT_WIDTH alu_in1, alu_in2, alu_out;
reg `BIT_WIDTH alu_start;
reg `BIT_WIDTH alu_done;

wire alu_start_set;
reg alu_done_set;
wire alu_done_rden;
parameter [3:0] ALU_STATE_OFF  = 4'd1;
parameter [3:0] ALU_STATE_ON   = 4'd2;
reg [3:0] alu_state;
reg `BIT_WIDTH alu_counter;
//------------------------------------------------


// MMIO control circuit
always@(negedge clk or posedge rst) begin
	if (rst) begin
	end
	else if (`MMIO_rden) begin
		//------------------------------------------------
		// mm_alu
		if (`MMIO_ALU_in1_rden) begin
			MMIODataBus <= alu_in1;
		end
		else if (`MMIO_ALU_in2_rden) begin
			MMIODataBus <= alu_in2;
		end
		else if (`MMIO_ALU_out_rden) begin
			MMIODataBus <= alu_out;
		end
		else if (`MMIO_ALU_start_rden) begin
			MMIODataBus <= alu_start;
		end
		else if (`MMIO_ALU_done_rden) begin
			MMIODataBus <= alu_done;
		end
		// add other MM regs to read from
	end
	else if (`MMIO_wren) begin
		//------------------------------------------------
		// mm_alu
		if (`MMIO_ALU_in1_wren)begin
			alu_in1 <= CPUDataBusOut;
		end
		else if (`MMIO_ALU_in2_wren)begin
			alu_in2 <= CPUDataBusOut;
		end
		// add other MM regs to write to
	end
end

//----------------------------------------------------------------------------------------
// alu_start: reset after set
always@(posedge clk or posedge rst) begin
	if (rst) begin
		alu_start <= 0;
	end
	else if (alu_start_set) begin
		alu_start <= 128'd1;
	end
	else if (alu_start == 128'd1) begin
		alu_start <= 0;
	end
end

// alu_done : reset after reading
always@(posedge clk or posedge rst) begin
	if (rst) begin
		alu_done <= 0;
	end
	else if (alu_done_set) begin
		alu_done <= 128'd1;
	end
	else if (alu_done_rden) begin
		alu_done <= 0;
	end
end

always@(posedge clk or posedge rst) begin
	if (rst) begin
		alu_state <= ALU_STATE_OFF;
		alu_counter <= 0;
		alu_done_set <= 0;
		alu_out <= 0;
	end
	else begin
		case(alu_state)
			ALU_STATE_OFF: begin
				alu_counter <= 0;
				alu_done_set <= 0;
				if (alu_start == 128'd1) begin
					alu_state <= ALU_STATE_ON;
				end
			end
			ALU_STATE_ON: begin
                alu_out <= alu_in1 + alu_in2;
                alu_done_set <= 1'b1;
                alu_state <= ALU_STATE_OFF;
			end
		endcase
	end
end
assign alu_done_rden = `MMIO_ALU_done_rden;
assign alu_start_set = `MMIO_ALU_start_wren && (CPUDataBusOut == 128'd1);
//----------------------------------------------------------------------------------------


assign mmio_readdata = MMIODataBus;
assign clk = pcie_256_hip_avmm_coreclkout_clk;
assign CPUDataBusOut = mm_interconnect_0_onchip_memory2_s1_writedata;