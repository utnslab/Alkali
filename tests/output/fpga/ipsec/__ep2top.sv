module ep2top#()
(
	 input  wire clk, 
	 input  wire rst,
	//output ports BUF
	output wire [512-1:0] DECRYPT_REQ_0_tdata ,
	output wire [64-1:0] DECRYPT_REQ_0_tkeep ,
	output wire  DECRYPT_REQ_0_tlast ,
	output wire  DECRYPT_REQ_0_tvalid ,
	input wire  DECRYPT_REQ_0_tready,
	//output ports STRUCT
	output wire [272-1:0] DECRYPT_REQ_1_tdata ,
	output wire  DECRYPT_REQ_1_tvalid ,
	input wire  DECRYPT_REQ_1_tready,
	//output ports STRUCT
	output wire [184-1:0] DECRYPT_REQ_2_tdata ,
	output wire  DECRYPT_REQ_2_tvalid ,
	input wire  DECRYPT_REQ_2_tready,
	//output ports STRUCT
	output wire [112-1:0] DECRYPT_REQ_3_tdata ,
	output wire  DECRYPT_REQ_3_tvalid ,
	input wire  DECRYPT_REQ_3_tready,
	//output ports STRUCT
	output wire [64-1:0] DECRYPT_REQ_4_tdata ,
	output wire  DECRYPT_REQ_4_tvalid ,
	input wire  DECRYPT_REQ_4_tready,
	//input ports BUF
	input wire [512-1:0] NET_RECV_0_tdata ,
	input wire [64-1:0] NET_RECV_0_tkeep ,
	input wire  NET_RECV_0_tlast ,
	input wire  NET_RECV_0_tvalid ,
	output wire  NET_RECV_0_tready
);
//
 wire [512-1:0] NET_RECV_0_r0_tdata;
 wire [64-1:0] NET_RECV_0_r0_tkeep;
 wire  NET_RECV_0_r0_tvalid;
 wire  NET_RECV_0_r0_tready;
 wire  NET_RECV_0_r0_tlast;

//
 wire [512-1:0] DECRYPT_REQ_0_r0_tdata;
 wire [64-1:0] DECRYPT_REQ_0_r0_tkeep;
 wire  DECRYPT_REQ_0_r0_tvalid;
 wire  DECRYPT_REQ_0_r0_tready;
 wire  DECRYPT_REQ_0_r0_tlast;

//
 wire [272-1:0] DECRYPT_REQ_1_r0_tdata;
 wire  DECRYPT_REQ_1_r0_tvalid;
 wire  DECRYPT_REQ_1_r0_tready;

//
 wire [184-1:0] DECRYPT_REQ_2_r0_tdata;
 wire  DECRYPT_REQ_2_r0_tvalid;
 wire  DECRYPT_REQ_2_r0_tready;

//
 wire [112-1:0] DECRYPT_REQ_3_r0_tdata;
 wire  DECRYPT_REQ_3_r0_tvalid;
 wire  DECRYPT_REQ_3_r0_tready;

//
 wire [64-1:0] DECRYPT_REQ_4_r0_tdata;
 wire  DECRYPT_REQ_4_r0_tvalid;
 wire  DECRYPT_REQ_4_r0_tready;

__handler_NET_RECV_process_packet#(
)__handler_NET_RECV_process_packet_20(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.NET_RECV_0_tdata({NET_RECV_0_r0_tdata}),
	.NET_RECV_0_tkeep({NET_RECV_0_r0_tkeep}),
	.NET_RECV_0_tlast({NET_RECV_0_r0_tlast}),
	.NET_RECV_0_tvalid({NET_RECV_0_r0_tvalid}),
	.NET_RECV_0_tready({NET_RECV_0_r0_tready}),
	//
	.DECRYPT_REQ_0_tdata({DECRYPT_REQ_0_r0_tdata}),
	.DECRYPT_REQ_0_tkeep({DECRYPT_REQ_0_r0_tkeep}),
	.DECRYPT_REQ_0_tlast({DECRYPT_REQ_0_r0_tlast}),
	.DECRYPT_REQ_0_tvalid({DECRYPT_REQ_0_r0_tvalid}),
	.DECRYPT_REQ_0_tready({DECRYPT_REQ_0_r0_tready}),
	//
	.DECRYPT_REQ_1_tdata({DECRYPT_REQ_1_r0_tdata}),
	.DECRYPT_REQ_1_tvalid({DECRYPT_REQ_1_r0_tvalid}),
	.DECRYPT_REQ_1_tready({DECRYPT_REQ_1_r0_tready}),
	//
	.DECRYPT_REQ_2_tdata({DECRYPT_REQ_2_r0_tdata}),
	.DECRYPT_REQ_2_tvalid({DECRYPT_REQ_2_r0_tvalid}),
	.DECRYPT_REQ_2_tready({DECRYPT_REQ_2_r0_tready}),
	//
	.DECRYPT_REQ_3_tdata({DECRYPT_REQ_3_r0_tdata}),
	.DECRYPT_REQ_3_tvalid({DECRYPT_REQ_3_r0_tvalid}),
	.DECRYPT_REQ_3_tready({DECRYPT_REQ_3_r0_tready}),
	//
	.DECRYPT_REQ_4_tdata({DECRYPT_REQ_4_r0_tdata}),
	.DECRYPT_REQ_4_tvalid({DECRYPT_REQ_4_r0_tvalid}),
	.DECRYPT_REQ_4_tready({DECRYPT_REQ_4_r0_tready})
);

//
 wire [512-1:0] in_0_DECRYPT_REQ_0_tdata;
 wire [64-1:0] in_0_DECRYPT_REQ_0_tkeep;
 wire  in_0_DECRYPT_REQ_0_tvalid;
 wire  in_0_DECRYPT_REQ_0_tready;
 wire  in_0_DECRYPT_REQ_0_tlast;

//
 wire [272-1:0] in_0_DECRYPT_REQ_1_tdata;
 wire  in_0_DECRYPT_REQ_1_tvalid;
 wire  in_0_DECRYPT_REQ_1_tready;

//
 wire [184-1:0] in_0_DECRYPT_REQ_2_tdata;
 wire  in_0_DECRYPT_REQ_2_tvalid;
 wire  in_0_DECRYPT_REQ_2_tready;

//
 wire [112-1:0] in_0_DECRYPT_REQ_3_tdata;
 wire  in_0_DECRYPT_REQ_3_tvalid;
 wire  in_0_DECRYPT_REQ_3_tready;

//
 wire [64-1:0] in_0_DECRYPT_REQ_4_tdata;
 wire  in_0_DECRYPT_REQ_4_tvalid;
 wire  in_0_DECRYPT_REQ_4_tready;

__controller_DECRYPT_REQ#(
)__controller_DECRYPT_REQ_21(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.in_0_DECRYPT_REQ_0_tdata({in_0_DECRYPT_REQ_0_tdata}),
	.in_0_DECRYPT_REQ_0_tkeep({in_0_DECRYPT_REQ_0_tkeep}),
	.in_0_DECRYPT_REQ_0_tlast({in_0_DECRYPT_REQ_0_tlast}),
	.in_0_DECRYPT_REQ_0_tvalid({in_0_DECRYPT_REQ_0_tvalid}),
	.in_0_DECRYPT_REQ_0_tready({in_0_DECRYPT_REQ_0_tready}),
	//
	.in_0_DECRYPT_REQ_1_tdata({in_0_DECRYPT_REQ_1_tdata}),
	.in_0_DECRYPT_REQ_1_tvalid({in_0_DECRYPT_REQ_1_tvalid}),
	.in_0_DECRYPT_REQ_1_tready({in_0_DECRYPT_REQ_1_tready}),
	//
	.in_0_DECRYPT_REQ_2_tdata({in_0_DECRYPT_REQ_2_tdata}),
	.in_0_DECRYPT_REQ_2_tvalid({in_0_DECRYPT_REQ_2_tvalid}),
	.in_0_DECRYPT_REQ_2_tready({in_0_DECRYPT_REQ_2_tready}),
	//
	.in_0_DECRYPT_REQ_3_tdata({in_0_DECRYPT_REQ_3_tdata}),
	.in_0_DECRYPT_REQ_3_tvalid({in_0_DECRYPT_REQ_3_tvalid}),
	.in_0_DECRYPT_REQ_3_tready({in_0_DECRYPT_REQ_3_tready}),
	//
	.in_0_DECRYPT_REQ_4_tdata({in_0_DECRYPT_REQ_4_tdata}),
	.in_0_DECRYPT_REQ_4_tvalid({in_0_DECRYPT_REQ_4_tvalid}),
	.in_0_DECRYPT_REQ_4_tready({in_0_DECRYPT_REQ_4_tready}),
	//
	.DECRYPT_REQ_0_tdata({DECRYPT_REQ_0_tdata}),
	.DECRYPT_REQ_0_tkeep({DECRYPT_REQ_0_tkeep}),
	.DECRYPT_REQ_0_tlast({DECRYPT_REQ_0_tlast}),
	.DECRYPT_REQ_0_tvalid({DECRYPT_REQ_0_tvalid}),
	.DECRYPT_REQ_0_tready({DECRYPT_REQ_0_tready}),
	//
	.DECRYPT_REQ_1_tdata({DECRYPT_REQ_1_tdata}),
	.DECRYPT_REQ_1_tvalid({DECRYPT_REQ_1_tvalid}),
	.DECRYPT_REQ_1_tready({DECRYPT_REQ_1_tready}),
	//
	.DECRYPT_REQ_2_tdata({DECRYPT_REQ_2_tdata}),
	.DECRYPT_REQ_2_tvalid({DECRYPT_REQ_2_tvalid}),
	.DECRYPT_REQ_2_tready({DECRYPT_REQ_2_tready}),
	//
	.DECRYPT_REQ_3_tdata({DECRYPT_REQ_3_tdata}),
	.DECRYPT_REQ_3_tvalid({DECRYPT_REQ_3_tvalid}),
	.DECRYPT_REQ_3_tready({DECRYPT_REQ_3_tready}),
	//
	.DECRYPT_REQ_4_tdata({DECRYPT_REQ_4_tdata}),
	.DECRYPT_REQ_4_tvalid({DECRYPT_REQ_4_tvalid}),
	.DECRYPT_REQ_4_tready({DECRYPT_REQ_4_tready}),
	//
	.DECRYPT_REQ_0_tdata({DECRYPT_REQ_0_tdata}),
	.DECRYPT_REQ_0_tkeep({DECRYPT_REQ_0_tkeep}),
	.DECRYPT_REQ_0_tlast({DECRYPT_REQ_0_tlast}),
	.DECRYPT_REQ_0_tvalid({DECRYPT_REQ_0_tvalid}),
	.DECRYPT_REQ_0_tready({DECRYPT_REQ_0_tready}),
	//
	.DECRYPT_REQ_1_tdata({DECRYPT_REQ_1_tdata}),
	.DECRYPT_REQ_1_tvalid({DECRYPT_REQ_1_tvalid}),
	.DECRYPT_REQ_1_tready({DECRYPT_REQ_1_tready}),
	//
	.DECRYPT_REQ_2_tdata({DECRYPT_REQ_2_tdata}),
	.DECRYPT_REQ_2_tvalid({DECRYPT_REQ_2_tvalid}),
	.DECRYPT_REQ_2_tready({DECRYPT_REQ_2_tready}),
	//
	.DECRYPT_REQ_3_tdata({DECRYPT_REQ_3_tdata}),
	.DECRYPT_REQ_3_tvalid({DECRYPT_REQ_3_tvalid}),
	.DECRYPT_REQ_3_tready({DECRYPT_REQ_3_tready}),
	//
	.DECRYPT_REQ_4_tdata({DECRYPT_REQ_4_tdata}),
	.DECRYPT_REQ_4_tvalid({DECRYPT_REQ_4_tvalid}),
	.DECRYPT_REQ_4_tready({DECRYPT_REQ_4_tready}),
	//
	.DECRYPT_REQ_0_tdata({DECRYPT_REQ_0_tdata}),
	.DECRYPT_REQ_0_tkeep({DECRYPT_REQ_0_tkeep}),
	.DECRYPT_REQ_0_tlast({DECRYPT_REQ_0_tlast}),
	.DECRYPT_REQ_0_tvalid({DECRYPT_REQ_0_tvalid}),
	.DECRYPT_REQ_0_tready({DECRYPT_REQ_0_tready}),
	//
	.DECRYPT_REQ_1_tdata({DECRYPT_REQ_1_tdata}),
	.DECRYPT_REQ_1_tvalid({DECRYPT_REQ_1_tvalid}),
	.DECRYPT_REQ_1_tready({DECRYPT_REQ_1_tready}),
	//
	.DECRYPT_REQ_2_tdata({DECRYPT_REQ_2_tdata}),
	.DECRYPT_REQ_2_tvalid({DECRYPT_REQ_2_tvalid}),
	.DECRYPT_REQ_2_tready({DECRYPT_REQ_2_tready}),
	//
	.DECRYPT_REQ_3_tdata({DECRYPT_REQ_3_tdata}),
	.DECRYPT_REQ_3_tvalid({DECRYPT_REQ_3_tvalid}),
	.DECRYPT_REQ_3_tready({DECRYPT_REQ_3_tready}),
	//
	.DECRYPT_REQ_4_tdata({DECRYPT_REQ_4_tdata}),
	.DECRYPT_REQ_4_tvalid({DECRYPT_REQ_4_tvalid}),
	.DECRYPT_REQ_4_tready({DECRYPT_REQ_4_tready})
);

 assign in_0_DECRYPT_REQ_0_tdata = DECRYPT_REQ_0_r0_tdata;
 assign in_0_DECRYPT_REQ_0_tvalid = DECRYPT_REQ_0_r0_tvalid;
 assign DECRYPT_REQ_0_r0_tready = in_0_DECRYPT_REQ_0_tready;
 assign in_0_DECRYPT_REQ_0_tkeep = DECRYPT_REQ_0_r0_tkeep;
 assign in_0_DECRYPT_REQ_0_tlast = DECRYPT_REQ_0_r0_tlast;

 assign in_0_DECRYPT_REQ_1_tdata = DECRYPT_REQ_1_r0_tdata;
 assign in_0_DECRYPT_REQ_1_tvalid = DECRYPT_REQ_1_r0_tvalid;
 assign DECRYPT_REQ_1_r0_tready = in_0_DECRYPT_REQ_1_tready;

 assign in_0_DECRYPT_REQ_2_tdata = DECRYPT_REQ_2_r0_tdata;
 assign in_0_DECRYPT_REQ_2_tvalid = DECRYPT_REQ_2_r0_tvalid;
 assign DECRYPT_REQ_2_r0_tready = in_0_DECRYPT_REQ_2_tready;

 assign in_0_DECRYPT_REQ_3_tdata = DECRYPT_REQ_3_r0_tdata;
 assign in_0_DECRYPT_REQ_3_tvalid = DECRYPT_REQ_3_r0_tvalid;
 assign DECRYPT_REQ_3_r0_tready = in_0_DECRYPT_REQ_3_tready;

 assign in_0_DECRYPT_REQ_4_tdata = DECRYPT_REQ_4_r0_tdata;
 assign in_0_DECRYPT_REQ_4_tvalid = DECRYPT_REQ_4_r0_tvalid;
 assign DECRYPT_REQ_4_r0_tready = in_0_DECRYPT_REQ_4_tready;

 assign NET_RECV_0_r0_tdata = NET_RECV_0_tdata;
 assign NET_RECV_0_r0_tvalid = NET_RECV_0_tvalid;
 assign NET_RECV_0_tready = NET_RECV_0_r0_tready;
 assign NET_RECV_0_r0_tkeep = NET_RECV_0_tkeep;
 assign NET_RECV_0_r0_tlast = NET_RECV_0_tlast;


endmodule
