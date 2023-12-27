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
	output wire [512-1:0] DMA_WRITE_1_tdata ,
	output wire [64-1:0] DMA_WRITE_1_tkeep ,
	output wire  DMA_WRITE_1_tlast ,
	output wire  DMA_WRITE_1_tvalid ,
	input wire  DMA_WRITE_1_tready,
	//output ports STRUCT
	output wire [64-1:0] DMA_WRITE_2_tdata ,
	output wire  DMA_WRITE_2_tvalid ,
	input wire  DMA_WRITE_2_tready,
	//output ports BUF
	output wire [512-1:0] DMA_WRITE_3_tdata ,
	output wire [64-1:0] DMA_WRITE_3_tkeep ,
	output wire  DMA_WRITE_3_tlast ,
	output wire  DMA_WRITE_3_tvalid ,
	input wire  DMA_WRITE_3_tready,
	//output ports STRUCT
	output wire [184-1:0] DMA_WRITE_4_tdata ,
	output wire  DMA_WRITE_4_tvalid ,
	input wire  DMA_WRITE_4_tready,
	//output ports STRUCT
	output wire [112-1:0] DMA_WRITE_5_tdata ,
	output wire  DMA_WRITE_5_tvalid ,
	input wire  DMA_WRITE_5_tready,
	//output ports STRUCT
	output wire [160-1:0] DMA_WRITE_6_tdata ,
	output wire  DMA_WRITE_6_tvalid ,
	input wire  DMA_WRITE_6_tready,
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
	.outport_0_1_tready({arg_28_tready}),
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
	.DMA_WRITE_1_tdata({DMA_WRITE_1_tdata}),
	.DMA_WRITE_1_tkeep({DMA_WRITE_1_tkeep}),
	.DMA_WRITE_1_tlast({DMA_WRITE_1_tlast}),
	.DMA_WRITE_1_tvalid({DMA_WRITE_1_tvalid}),
	.DMA_WRITE_1_tready({DMA_WRITE_1_tready}),
	//
	.DMA_WRITE_2_tdata({DMA_WRITE_2_tdata}),
	.DMA_WRITE_2_tvalid({DMA_WRITE_2_tvalid}),
	.DMA_WRITE_2_tready({DMA_WRITE_2_tready}),
	//
	.DMA_WRITE_3_tdata({DMA_WRITE_3_tdata}),
	.DMA_WRITE_3_tkeep({DMA_WRITE_3_tkeep}),
	.DMA_WRITE_3_tlast({DMA_WRITE_3_tlast}),
	.DMA_WRITE_3_tvalid({DMA_WRITE_3_tvalid}),
	.DMA_WRITE_3_tready({DMA_WRITE_3_tready}),
	//
	.DMA_WRITE_4_tdata({DMA_WRITE_4_tdata}),
	.DMA_WRITE_4_tvalid({DMA_WRITE_4_tvalid}),
	.DMA_WRITE_4_tready({DMA_WRITE_4_tready}),
	//
	.DMA_WRITE_5_tdata({DMA_WRITE_5_tdata}),
	.DMA_WRITE_5_tvalid({DMA_WRITE_5_tvalid}),
	.DMA_WRITE_5_tready({DMA_WRITE_5_tready}),
	//
	.DMA_WRITE_6_tdata({DMA_WRITE_6_tdata}),
	.DMA_WRITE_6_tvalid({DMA_WRITE_6_tvalid}),
	.DMA_WRITE_6_tready({DMA_WRITE_6_tready}),
	//
	.outport_33_1_tdata({arg_151_tdata}),
	.outport_33_1_tvalid({arg_151_tvalid}),
	.outport_33_1_tready({arg_151_tready}),
	//
	.outport_33_2_tdata({arg_152_tdata}),
	.outport_33_2_tkeep({arg_152_tkeep}),
	.outport_33_2_tlast({arg_152_tlast}),
	.outport_33_2_tvalid({arg_152_tvalid}),
	.outport_33_2_tready({arg_152_tready}),
	//
	.outport_33_3_tdata({arg_153_tdata}),
	.outport_33_3_tvalid({arg_153_tvalid}),
	.outport_33_3_tready({arg_153_tready}),
	//
	.outport_33_4_tdata({arg_154_tdata}),
	.outport_33_4_tvalid({arg_154_tvalid}),
	.outport_33_4_tready({arg_154_tready}),
	//
	.outport_33_5_tdata({arg_155_tdata}),
	.outport_33_5_tvalid({arg_155_tvalid}),
	.outport_33_5_tready({arg_155_tready})
);

//
 wire [96-1:0] arg_151_tdata;
 wire  arg_151_tvalid;
 wire  arg_151_tready;

//
 wire [512-1:0] arg_152_tdata;
 wire [64-1:0] arg_152_tkeep;
 wire  arg_152_tvalid;
 wire  arg_152_tready;
 wire  arg_152_tlast;

//
 wire [184-1:0] arg_153_tdata;
 wire  arg_153_tvalid;
 wire  arg_153_tready;

//
 wire [112-1:0] arg_154_tdata;
 wire  arg_154_tvalid;
 wire  arg_154_tready;

//
 wire [160-1:0] arg_155_tdata;
 wire  arg_155_tvalid;
 wire  arg_155_tready;

__handler_ACK_GEN_ack_gen#(
)__handler_ACK_GEN_ack_gen(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.arg_151_tdata({arg_151_tdata}),
	.arg_151_tvalid({arg_151_tvalid}),
	.arg_151_tready({arg_151_tready}),
	//
	.arg_152_tdata({arg_152_tdata}),
	.arg_152_tkeep({arg_152_tkeep}),
	.arg_152_tlast({arg_152_tlast}),
	.arg_152_tvalid({arg_152_tvalid}),
	.arg_152_tready({arg_152_tready}),
	//
	.arg_153_tdata({arg_153_tdata}),
	.arg_153_tvalid({arg_153_tvalid}),
	.arg_153_tready({arg_153_tready}),
	//
	.arg_154_tdata({arg_154_tdata}),
	.arg_154_tvalid({arg_154_tvalid}),
	.arg_154_tready({arg_154_tready}),
	//
	.arg_155_tdata({arg_155_tdata}),
	.arg_155_tvalid({arg_155_tvalid}),
	.arg_155_tready({arg_155_tready}),
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


endmodule
