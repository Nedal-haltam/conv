// ============================================================================
// Copyright (c) 2017 by Terasic Technologies Inc.
// ============================================================================
//
// Permission:
//
//   Terasic grants permission to use and modify this code for use
//   in synthesis for all Terasic Development Boards and Altera Development 
//   Kits made by Terasic.  Other use of this code, including the selling 
//   ,duplication, or modification of any portion is strictly prohibited.
//
// Disclaimer:
//
//   This VHDL/Verilog or C/C++ source code is intended as a design reference
//   which illustrates how these types of functions can be implemented.
//   It is the user's responsibility to verify their design for
//   consistency and functionality through the use of formal
//   verification methods.  Terasic provides no warranty regarding the use 
//   or functionality of this code.
//
// ============================================================================
//           
//  Terasic Technologies Inc
//  9F., No.176, Sec.2, Gongdao 5th Rd, East Dist, Hsinchu City, 30070. Taiwan
//  
//  
//                     web: http://www.terasic.com/  
//                     email: support@terasic.com
//
// ============================================================================
//Date:  Tue Nov 21 13:54:58 2017
// ============================================================================

`define ENABLE_PCIE

module PCIe_Fundamental(

	//////////// CLOCK //////////
	input 		          		CLOCK_50_B3B,
	input 		          		CLOCK_50_B4A,
	input 		          		CLOCK_50_B5B,
	input 		          		CLOCK_50_B6A,
	input 		          		CLOCK_50_B7A,
	input 		          		CLOCK_50_B8A,

	//////////// Buttons //////////
	input 		          		CPU_RESET_n,
	input 		     [3:0]		KEY,

	//////////// Swtiches //////////
	input 		     [3:0]		SW,

	//////////// LED //////////
	output		     [3:0]		LED,

	//////////// HEX0 //////////
	output		     [6:0]		HEX0,
	output		          		HEX0_DP,

	//////////// HEX1 //////////
	output		     [6:0]		HEX1,
	output		          		HEX1_DP,

	//////////// FAN //////////
	output		          		FAN_CTRL,

	//////////// SDRAM //////////
	output		    [12:0]		DRAM_ADDR,
	output		     [1:0]		DRAM_BA,
	output		          		DRAM_CAS_n,
	output		          		DRAM_CKE,
	output		          		DRAM_CLK,
	output		          		DRAM_CS_n,
	inout 		    [15:0]		DRAM_DQ,
	output		          		DRAM_LDQM,
	output		          		DRAM_RAS_n,
	output		          		DRAM_UDQM,
	output		          		DRAM_WE_n,

	//////////// Uart to Usb //////////
	input 		          		UART_CTS,
	output		          		UART_RTS,
	input 		          		UART_RX,
	output		          		UART_TX,

	//////////// Arduino Interface //////////
	output		          		ADC_CONVST,
	output		          		ADC_SCK,
	output		          		ADC_SDI,
	input 		          		ADC_SDO,
	inout 		    [15:0]		ARD_IO,
	
`ifdef ENABLE_PCIE
	//////////// PCIE //////////
	inout 		          		PCIE_PERST_n,
	input 		          		PCIE_REFCLK_p,
	input 		     [3:0]		PCIE_RX_p,
	inout 		          		PCIE_SMBCLK,
	inout 		          		PCIE_SMBDAT,
	output		     [3:0]		PCIE_TX_p,
	inout 		          		PCIE_WAKE_n,
`endif /*ENABLE_PCIE*/

	//////////// SMA //////////
	input 		          		SMA_CLKIN,
	output		          		SMA_CLKOUT
);


//=======================================================
//  REG/WIRE declarations
//=======================================================
wire [3:0]	pio_led;
wire [3:0]	pio_button;
wire [31:0] pcie_hip_ctrl_test_in;
//////////////////////
// PCIE RESET
wire     any_rstn;
reg      any_rstn_r  /* synthesis ALTERA_ATTRIBUTE = "SUPPRESS_DA_RULE_INTERNAL=R102"  */;
reg      any_rstn_rr /* synthesis ALTERA_ATTRIBUTE = "SUPPRESS_DA_RULE_INTERNAL=R102"  */;
//=======================================================
//  Structural coding
//=======================================================
assign 	pcie_hip_ctrl_test_in[4:0]  =  5'b01000;
assign 	pcie_hip_ctrl_test_in[5]    =  1'b1;
assign 	pcie_hip_ctrl_test_in[31:6] =  26'h2;

assign 	FAN_CTRL    = 1'b1;
assign 	PCIE_WAKE_n = 1'b1;
assign 	any_rstn    = PCIE_PERST_n & CPU_RESET_n ;
  
//reset Synchronizer
always @(posedge CLOCK_50_B3B or negedge any_rstn)
    begin
      if (any_rstn == 0)
        begin
          any_rstn_r <= 0;
          any_rstn_rr <= 0;
        end
      else
        begin
          any_rstn_r <= 1;
          any_rstn_rr <= any_rstn_r;
        end
    end


top u0 (
		.clk_clk                                  (CLOCK_50_B3B),                                                              //                                   clk.clk
		.reset_reset_n                            ((CPU_RESET_n == 1'b0) ? 1'b0 : (PCIE_PERST_n == 1'b0) ? 1'b0 : 1'b1),       //                                 reset.reset_n

		.pio_led_external_connection_export       (pio_led),                                                                   //           pio_led_external_connection.export
		.pio_button_external_connection_export    (pio_button),                                                                //        pio_button_external_connection.export

		.refclk_clk                          		(PCIE_REFCLK_p),                                                             //              pcie_256_hip_avmm_refclk.clk
		.pcie_rstn_npor                           (any_rstn_rr),                                                               //                pcie_256_hip_avmm_npor.npor
		.pcie_rstn_pin_perst                      (PCIE_PERST_n),                                                              //                                      .pin_perst

		.hip_serial_rx_in0                   		(PCIE_RX_p[0]),                                                              //          pcie_256_hip_avmm_hip_serial.rx_in0
		.hip_serial_rx_in1                   		(PCIE_RX_p[1]),                                                              //                                      .rx_in1
		.hip_serial_rx_in2                   		(PCIE_RX_p[2]),                                                              //                                      .rx_in2
		.hip_serial_rx_in3                   		(PCIE_RX_p[3]),                                                              //                                      .rx_in3
		.hip_serial_tx_out0                  		(PCIE_TX_p[0]),                                                              //                                      .tx_out0
		.hip_serial_tx_out1                  		(PCIE_TX_p[1]),                                                              //                                      .tx_out1
		.hip_serial_tx_out2                  		(PCIE_TX_p[2]),                                                              //                                      .tx_out2
		.hip_serial_tx_out3                  		(PCIE_TX_p[3]),                                                              //                                      .tx_out3

		.hip_pipe_sim_pipe_pclk_in           		(1'b0),                                                                      //            pcie_256_hip_avmm_hip_pipe.sim_pipe_pclk_in
		.hip_pipe_sim_pipe_rate              		(1'b0),                                                                      //                                      .sim_pipe_rate
		.hip_ctrl_test_in                    		(pcie_hip_ctrl_test_in),                                                     //            pcie_256_hip_avmm_hip_ctrl.test_in
		.hip_ctrl_simu_mode_pipe             		(1'b0),                                                                       //                                      .simu_mode_pipe
		.rst(~KEY[0])
	);
	

assign 	LED        = pio_led;
assign 	pio_button = KEY;		// button low-active

endmodule
