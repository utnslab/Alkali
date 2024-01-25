module __handler_DECRYPT_REQ_crypto#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports BUF
	input wire [512-1:0] arg_0_tdata ,
	input wire [64-1:0] arg_0_tkeep ,
	input wire  arg_0_tlast ,
	input wire  arg_0_tvalid ,
	output wire  arg_0_tready,
	//input ports STRUCT
	input wire [272-1:0] arg_1_tdata ,
	input wire  arg_1_tvalid ,
	output wire  arg_1_tready,
	//input ports STRUCT
	input wire [184-1:0] arg_2_tdata ,
	input wire  arg_2_tvalid ,
	output wire  arg_2_tready,
	//input ports STRUCT
	input wire [112-1:0] arg_3_tdata ,
	input wire  arg_3_tvalid ,
	output wire  arg_3_tready,
	//input ports STRUCT
	input wire [64-1:0] arg_4_tdata ,
	input wire  arg_4_tvalid ,
	output wire  arg_4_tready
);

endmodule
