module ep2top#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports BUF
	input wire [512-1:0] NET_RECV_0_tdata ,
	input wire [64-1:0] NET_RECV_0_tkeep ,
	input wire  NET_RECV_0_tlast ,
	input wire  NET_RECV_0_tvalid ,
	output wire  NET_RECV_0_tready,
	//output ports BUF
	output wire [512-1:0] NET_SEND_0_tdata ,
	output wire [64-1:0] NET_SEND_0_tkeep ,
	output wire  NET_SEND_0_tlast ,
	output wire  NET_SEND_0_tvalid ,
	input wire  NET_SEND_0_tready,
	//output ports BUF
	output wire [512-1:0] NET_SEND_1_tdata ,
	output wire [64-1:0] NET_SEND_1_tkeep ,
	output wire  NET_SEND_1_tlast ,
	output wire  NET_SEND_1_tvalid ,
	input wire  NET_SEND_1_tready,
	//output ports STRUCT
	output wire [184-1:0] NET_SEND_2_tdata ,
	output wire  NET_SEND_2_tvalid ,
	input wire  NET_SEND_2_tready,
	//output ports STRUCT
	output wire [112-1:0] NET_SEND_3_tdata ,
	output wire  NET_SEND_3_tvalid ,
	input wire  NET_SEND_3_tready,
	//output ports STRUCT
	output wire [160-1:0] NET_SEND_4_tdata ,
	output wire  NET_SEND_4_tvalid ,
	input wire  NET_SEND_4_tready
);
__handler_NET_RECV_process_packet#(
)__handler_NET_RECV_process_packet(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.NET_RECV_0_tdata({NET_RECV_0_tdata}),
	.NET_RECV_0_tkeep({NET_RECV_0_tkeep}),
	.NET_RECV_0_tlast({NET_RECV_0_tlast}),
	.NET_RECV_0_tvalid({NET_RECV_0_tvalid}),
	.NET_RECV_0_tready({NET_RECV_0_tready}),
	//
	.outport_0_1_tdata({arg_28_tdata}),
	.outport_0_1_tvalid({arg_28_tvalid}),
	.outport_0_1_tready({1}),
	//
	.outport_0_2_tdata({arg_29_tdata}),
	.outport_0_2_tkeep({arg_29_tkeep}),
	.outport_0_2_tlast({arg_29_tlast}),
	.outport_0_2_tvalid({arg_29_tvalid}),
	.outport_0_2_tready({arg_29_tready}),
	//
	.outport_0_3_tdata({arg_30_tdata}),
	.outport_0_3_tvalid({arg_30_tvalid}),
	.outport_0_3_tready({arg_30_tready}),
	//
	.outport_0_4_tdata({arg_31_tdata}),
	.outport_0_4_tvalid({arg_31_tvalid}),
	.outport_0_4_tready({arg_31_tready}),
	//
	.outport_0_5_tdata({arg_32_tdata}),
	.outport_0_5_tvalid({arg_32_tvalid}),
	.outport_0_5_tready({arg_32_tready})
);

//
 wire [96-1:0] arg_126_tdata;
 wire  arg_126_tvalid;
 wire  arg_126_tready;

//
 wire [512-1:0] arg_127_tdata;
 wire [64-1:0] arg_127_tkeep;
 wire  arg_127_tvalid;
 wire  arg_127_tready;
 wire  arg_127_tlast;

//
 wire [184-1:0] arg_128_tdata;
 wire  arg_128_tvalid;
 wire  arg_128_tready;

//
 wire [112-1:0] arg_129_tdata;
 wire  arg_129_tvalid;
 wire  arg_129_tready;

//
 wire [160-1:0] arg_130_tdata;
 wire  arg_130_tvalid;
 wire  arg_130_tready;

__handler_ACK_GEN_ack_gen#(
)__handler_ACK_GEN_ack_gen(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.arg_126_tdata({arg_126_tdata}),
	.arg_126_tvalid({arg_126_tvalid}),
	.arg_126_tready({arg_126_tready}),
	//
	.arg_127_tdata({arg_127_tdata}),
	.arg_127_tkeep({arg_127_tkeep}),
	.arg_127_tlast({arg_127_tlast}),
	.arg_127_tvalid({arg_127_tvalid}),
	.arg_127_tready({arg_127_tready}),
	//
	.arg_128_tdata({arg_128_tdata}),
	.arg_128_tvalid({arg_128_tvalid}),
	.arg_128_tready({arg_128_tready}),
	//
	.arg_129_tdata({arg_129_tdata}),
	.arg_129_tvalid({arg_129_tvalid}),
	.arg_129_tready({arg_129_tready}),
	//
	.arg_130_tdata({arg_130_tdata}),
	.arg_130_tvalid({arg_130_tvalid}),
	.arg_130_tready({arg_130_tready}),
	//
	.NET_SEND_0_tdata({NET_SEND_0_tdata}),
	.NET_SEND_0_tkeep({NET_SEND_0_tkeep}),
	.NET_SEND_0_tlast({NET_SEND_0_tlast}),
	.NET_SEND_0_tvalid({NET_SEND_0_tvalid}),
	.NET_SEND_0_tready({NET_SEND_0_tready}),
	//
	.NET_SEND_1_tdata({NET_SEND_1_tdata}),
	.NET_SEND_1_tkeep({NET_SEND_1_tkeep}),
	.NET_SEND_1_tlast({NET_SEND_1_tlast}),
	.NET_SEND_1_tvalid({NET_SEND_1_tvalid}),
	.NET_SEND_1_tready({NET_SEND_1_tready}),
	//
	.NET_SEND_2_tdata({NET_SEND_2_tdata}),
	.NET_SEND_2_tvalid({NET_SEND_2_tvalid}),
	.NET_SEND_2_tready({NET_SEND_2_tready}),
	//
	.NET_SEND_3_tdata({NET_SEND_3_tdata}),
	.NET_SEND_3_tvalid({NET_SEND_3_tvalid}),
	.NET_SEND_3_tready({NET_SEND_3_tready}),
	//
	.NET_SEND_4_tdata({NET_SEND_4_tdata}),
	.NET_SEND_4_tvalid({NET_SEND_4_tvalid}),
	.NET_SEND_4_tready({NET_SEND_4_tready})
);

//
 wire [96-1:0] arg_28_tdata;
 wire  arg_28_tvalid;
 wire  arg_28_tready;

//
 wire [512-1:0] arg_29_tdata;
 wire [64-1:0] arg_29_tkeep;
 wire  arg_29_tvalid;
 wire  arg_29_tready;
 wire  arg_29_tlast;

//
 wire [184-1:0] arg_30_tdata;
 wire  arg_30_tvalid;
 wire  arg_30_tready;

//
 wire [112-1:0] arg_31_tdata;
 wire  arg_31_tvalid;
 wire  arg_31_tready;

//
 wire [160-1:0] arg_32_tdata;
 wire  arg_32_tvalid;
 wire  arg_32_tready;

__handler_OoO_DETECT_OoO_detection#(
)__handler_OoO_DETECT_OoO_detection(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.arg_28_tdata({arg_28_tdata}),
	.arg_28_tvalid({arg_28_tvalid}),
	.arg_28_tready({arg_28_tready}),
	//
	.arg_29_tdata({arg_29_tdata}),
	.arg_29_tkeep({arg_29_tkeep}),
	.arg_29_tlast({arg_29_tlast}),
	.arg_29_tvalid({arg_29_tvalid}),
	.arg_29_tready({arg_29_tready}),
	//
	.arg_30_tdata({arg_30_tdata}),
	.arg_30_tvalid({arg_30_tvalid}),
	.arg_30_tready({arg_30_tready}),
	//
	.arg_31_tdata({arg_31_tdata}),
	.arg_31_tvalid({arg_31_tvalid}),
	.arg_31_tready({arg_31_tready}),
	//
	.arg_32_tdata({arg_32_tdata}),
	.arg_32_tvalid({arg_32_tvalid}),
	.arg_32_tready({arg_32_tready}),
	//
	.outport_33_1_tdata({arg_126_tdata}),
	.outport_33_1_tvalid({arg_126_tvalid}),
	.outport_33_1_tready({arg_126_tready}),
	//
	.outport_33_2_tdata({arg_127_tdata}),
	.outport_33_2_tkeep({arg_127_tkeep}),
	.outport_33_2_tlast({arg_127_tlast}),
	.outport_33_2_tvalid({arg_127_tvalid}),
	.outport_33_2_tready({arg_127_tready}),
	//
	.outport_33_3_tdata({arg_128_tdata}),
	.outport_33_3_tvalid({arg_128_tvalid}),
	.outport_33_3_tready({arg_128_tready}),
	//
	.outport_33_4_tdata({arg_129_tdata}),
	.outport_33_4_tvalid({arg_129_tvalid}),
	.outport_33_4_tready({arg_129_tready}),
	//
	.outport_33_5_tdata({arg_130_tdata}),
	.outport_33_5_tvalid({arg_130_tvalid}),
	.outport_33_5_tready({arg_130_tready})
);


endmodule
