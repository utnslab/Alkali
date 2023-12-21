module ep2top#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports BUF
	input wire [512-1:0] NET_RECV_1_tdata ,
	input wire [64-1:0] NET_RECV_1_tkeep ,
	input wire  NET_RECV_1_tlast ,
	input wire  NET_RECV_1_tvalid ,
	output wire  NET_RECV_1_tready
);
__handler_NET_RECV_process_packet#(
)__handler_NET_RECV_process_packet(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.NET_RECV_1_tdata({NET_RECV_1_tdata}),
	.NET_RECV_1_tkeep({NET_RECV_1_tkeep}),
	.NET_RECV_1_tlast({NET_RECV_1_tlast}),
	.NET_RECV_1_tvalid({NET_RECV_1_tvalid}),
	.NET_RECV_1_tready({NET_RECV_1_tready}),
	//
	.outport_3_2_tdata({arg_19_tdata}),
	.outport_3_2_tvalid({arg_19_tvalid}),
	.outport_3_2_tready({arg_19_tready})
);

//
 wire [96-1:0] arg_19_tdata;
 wire  arg_19_tvalid;
 wire  arg_19_tready;

__handler_OoO_DETECT_ooo_detect#(
)__handler_OoO_DETECT_ooo_detect(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.arg_19_tdata({arg_19_tdata}),
	.arg_19_tvalid({arg_19_tvalid}),
	.arg_19_tready({arg_19_tready})
);


endmodule
