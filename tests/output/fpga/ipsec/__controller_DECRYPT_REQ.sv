module __controller_DECRYPT_REQ#()
(
	 input  wire clk, 
	 input  wire rst,
	//
	input wire [512-1:0] in_0_DECRYPT_REQ_0_tdata ,
	input wire [64-1:0] in_0_DECRYPT_REQ_0_tkeep ,
	input wire  in_0_DECRYPT_REQ_0_tlast ,
	input wire  in_0_DECRYPT_REQ_0_tvalid ,
	output wire  in_0_DECRYPT_REQ_0_tready,
	//
	input wire [272-1:0] in_0_DECRYPT_REQ_1_tdata ,
	input wire  in_0_DECRYPT_REQ_1_tvalid ,
	output wire  in_0_DECRYPT_REQ_1_tready,
	//
	input wire [184-1:0] in_0_DECRYPT_REQ_2_tdata ,
	input wire  in_0_DECRYPT_REQ_2_tvalid ,
	output wire  in_0_DECRYPT_REQ_2_tready,
	//
	input wire [112-1:0] in_0_DECRYPT_REQ_3_tdata ,
	input wire  in_0_DECRYPT_REQ_3_tvalid ,
	output wire  in_0_DECRYPT_REQ_3_tready,
	//
	input wire [64-1:0] in_0_DECRYPT_REQ_4_tdata ,
	input wire  in_0_DECRYPT_REQ_4_tvalid ,
	output wire  in_0_DECRYPT_REQ_4_tready,
	//
	output wire [512-1:0] DECRYPT_REQ_0_tdata ,
	output wire [64-1:0] DECRYPT_REQ_0_tkeep ,
	output wire  DECRYPT_REQ_0_tlast ,
	output wire  DECRYPT_REQ_0_tvalid ,
	input wire  DECRYPT_REQ_0_tready,
	//
	output wire [272-1:0] DECRYPT_REQ_1_tdata ,
	output wire  DECRYPT_REQ_1_tvalid ,
	input wire  DECRYPT_REQ_1_tready,
	//
	output wire [184-1:0] DECRYPT_REQ_2_tdata ,
	output wire  DECRYPT_REQ_2_tvalid ,
	input wire  DECRYPT_REQ_2_tready,
	//
	output wire [112-1:0] DECRYPT_REQ_3_tdata ,
	output wire  DECRYPT_REQ_3_tvalid ,
	input wire  DECRYPT_REQ_3_tready,
	//
	output wire [64-1:0] DECRYPT_REQ_4_tdata ,
	output wire  DECRYPT_REQ_4_tvalid ,
	input wire  DECRYPT_REQ_4_tready,
	//
	output wire [512-1:0] DECRYPT_REQ_0_tdata ,
	output wire [64-1:0] DECRYPT_REQ_0_tkeep ,
	output wire  DECRYPT_REQ_0_tlast ,
	output wire  DECRYPT_REQ_0_tvalid ,
	input wire  DECRYPT_REQ_0_tready,
	//
	output wire [272-1:0] DECRYPT_REQ_1_tdata ,
	output wire  DECRYPT_REQ_1_tvalid ,
	input wire  DECRYPT_REQ_1_tready,
	//
	output wire [184-1:0] DECRYPT_REQ_2_tdata ,
	output wire  DECRYPT_REQ_2_tvalid ,
	input wire  DECRYPT_REQ_2_tready,
	//
	output wire [112-1:0] DECRYPT_REQ_3_tdata ,
	output wire  DECRYPT_REQ_3_tvalid ,
	input wire  DECRYPT_REQ_3_tready,
	//
	output wire [64-1:0] DECRYPT_REQ_4_tdata ,
	output wire  DECRYPT_REQ_4_tvalid ,
	input wire  DECRYPT_REQ_4_tready,
	//
	output wire [512-1:0] DECRYPT_REQ_0_tdata ,
	output wire [64-1:0] DECRYPT_REQ_0_tkeep ,
	output wire  DECRYPT_REQ_0_tlast ,
	output wire  DECRYPT_REQ_0_tvalid ,
	input wire  DECRYPT_REQ_0_tready,
	//
	output wire [272-1:0] DECRYPT_REQ_1_tdata ,
	output wire  DECRYPT_REQ_1_tvalid ,
	input wire  DECRYPT_REQ_1_tready,
	//
	output wire [184-1:0] DECRYPT_REQ_2_tdata ,
	output wire  DECRYPT_REQ_2_tvalid ,
	input wire  DECRYPT_REQ_2_tready,
	//
	output wire [112-1:0] DECRYPT_REQ_3_tdata ,
	output wire  DECRYPT_REQ_3_tvalid ,
	input wire  DECRYPT_REQ_3_tready,
	//
	output wire [64-1:0] DECRYPT_REQ_4_tdata ,
	output wire  DECRYPT_REQ_4_tvalid ,
	input wire  DECRYPT_REQ_4_tready
);
ctrl_demux#(
.D_COUNT  (3),
.DATA_WIDTH(512),
.KEEP_ENABLE (1)
)ctrl_demux_11(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata({in_0_DECRYPT_REQ_0_tdata}),
	.s_val_axis_tkeep({in_0_DECRYPT_REQ_0_tkeep}),
	.s_val_axis_tlast({in_0_DECRYPT_REQ_0_tlast}),
	.s_val_axis_tvalid({in_0_DECRYPT_REQ_0_tvalid}),
	.s_val_axis_tready({in_0_DECRYPT_REQ_0_tready}),
	//(des)mux out
	.m_val_axis_tdata({DECRYPT_REQ_0_tdata,DECRYPT_REQ_0_tdata,DECRYPT_REQ_0_tdata}),
	.m_val_axis_tkeep({DECRYPT_REQ_0_tkeep,DECRYPT_REQ_0_tkeep,DECRYPT_REQ_0_tkeep}),
	.m_val_axis_tlast({DECRYPT_REQ_0_tlast,DECRYPT_REQ_0_tlast,DECRYPT_REQ_0_tlast}),
	.m_val_axis_tvalid({DECRYPT_REQ_0_tvalid,DECRYPT_REQ_0_tvalid,DECRYPT_REQ_0_tvalid}),
	.m_val_axis_tready({DECRYPT_REQ_0_tready,DECRYPT_REQ_0_tready,DECRYPT_REQ_0_tready}),
	//selector wire
	.s_selector_tdata({ctrl_demux_select_10_tdata}),
	.s_selector_tvalid({ctrl_demux_select_10_tvalid}),
	.s_selector_tready({ctrl_demux_select_10_tready})
);

ctrl_demux#(
.D_COUNT  (3),
.DATA_WIDTH(272),
.KEEP_ENABLE (0)
)ctrl_demux_13(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata({in_0_DECRYPT_REQ_1_tdata}),
	.s_val_axis_tvalid({in_0_DECRYPT_REQ_1_tvalid}),
	.s_val_axis_tready({in_0_DECRYPT_REQ_1_tready}),
	//(des)mux out
	.m_val_axis_tdata({DECRYPT_REQ_1_tdata,DECRYPT_REQ_1_tdata,DECRYPT_REQ_1_tdata}),
	.m_val_axis_tvalid({DECRYPT_REQ_1_tvalid,DECRYPT_REQ_1_tvalid,DECRYPT_REQ_1_tvalid}),
	.m_val_axis_tready({DECRYPT_REQ_1_tready,DECRYPT_REQ_1_tready,DECRYPT_REQ_1_tready}),
	//selector wire
	.s_selector_tdata({ctrl_demux_select_12_tdata}),
	.s_selector_tvalid({ctrl_demux_select_12_tvalid}),
	.s_selector_tready({ctrl_demux_select_12_tready})
);

ctrl_demux#(
.D_COUNT  (3),
.DATA_WIDTH(184),
.KEEP_ENABLE (0)
)ctrl_demux_15(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata({in_0_DECRYPT_REQ_2_tdata}),
	.s_val_axis_tvalid({in_0_DECRYPT_REQ_2_tvalid}),
	.s_val_axis_tready({in_0_DECRYPT_REQ_2_tready}),
	//(des)mux out
	.m_val_axis_tdata({DECRYPT_REQ_2_tdata,DECRYPT_REQ_2_tdata,DECRYPT_REQ_2_tdata}),
	.m_val_axis_tvalid({DECRYPT_REQ_2_tvalid,DECRYPT_REQ_2_tvalid,DECRYPT_REQ_2_tvalid}),
	.m_val_axis_tready({DECRYPT_REQ_2_tready,DECRYPT_REQ_2_tready,DECRYPT_REQ_2_tready}),
	//selector wire
	.s_selector_tdata({ctrl_demux_select_14_tdata}),
	.s_selector_tvalid({ctrl_demux_select_14_tvalid}),
	.s_selector_tready({ctrl_demux_select_14_tready})
);

ctrl_demux#(
.D_COUNT  (3),
.DATA_WIDTH(112),
.KEEP_ENABLE (0)
)ctrl_demux_17(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata({in_0_DECRYPT_REQ_3_tdata}),
	.s_val_axis_tvalid({in_0_DECRYPT_REQ_3_tvalid}),
	.s_val_axis_tready({in_0_DECRYPT_REQ_3_tready}),
	//(des)mux out
	.m_val_axis_tdata({DECRYPT_REQ_3_tdata,DECRYPT_REQ_3_tdata,DECRYPT_REQ_3_tdata}),
	.m_val_axis_tvalid({DECRYPT_REQ_3_tvalid,DECRYPT_REQ_3_tvalid,DECRYPT_REQ_3_tvalid}),
	.m_val_axis_tready({DECRYPT_REQ_3_tready,DECRYPT_REQ_3_tready,DECRYPT_REQ_3_tready}),
	//selector wire
	.s_selector_tdata({ctrl_demux_select_16_tdata}),
	.s_selector_tvalid({ctrl_demux_select_16_tvalid}),
	.s_selector_tready({ctrl_demux_select_16_tready})
);

ctrl_demux#(
.D_COUNT  (3),
.DATA_WIDTH(64),
.KEEP_ENABLE (0)
)ctrl_demux_19(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata({in_0_DECRYPT_REQ_4_tdata}),
	.s_val_axis_tvalid({in_0_DECRYPT_REQ_4_tvalid}),
	.s_val_axis_tready({in_0_DECRYPT_REQ_4_tready}),
	//(des)mux out
	.m_val_axis_tdata({DECRYPT_REQ_4_tdata,DECRYPT_REQ_4_tdata,DECRYPT_REQ_4_tdata}),
	.m_val_axis_tvalid({DECRYPT_REQ_4_tvalid,DECRYPT_REQ_4_tvalid,DECRYPT_REQ_4_tvalid}),
	.m_val_axis_tready({DECRYPT_REQ_4_tready,DECRYPT_REQ_4_tready,DECRYPT_REQ_4_tready}),
	//selector wire
	.s_selector_tdata({ctrl_demux_select_18_tdata}),
	.s_selector_tvalid({ctrl_demux_select_18_tvalid}),
	.s_selector_tready({ctrl_demux_select_18_tready})
);

//selector wire
 wire [2-1:0] ctrl_demux_select_10_tdata;
 wire  ctrl_demux_select_10_tvalid;
 wire  ctrl_demux_select_10_tready;

//selector wire
 wire [2-1:0] ctrl_demux_select_12_tdata;
 wire  ctrl_demux_select_12_tvalid;
 wire  ctrl_demux_select_12_tready;

//selector wire
 wire [2-1:0] ctrl_demux_select_14_tdata;
 wire  ctrl_demux_select_14_tvalid;
 wire  ctrl_demux_select_14_tready;

//selector wire
 wire [2-1:0] ctrl_demux_select_16_tdata;
 wire  ctrl_demux_select_16_tvalid;
 wire  ctrl_demux_select_16_tready;

//selector wire
 wire [2-1:0] ctrl_demux_select_18_tdata;
 wire  ctrl_demux_select_18_tvalid;
 wire  ctrl_demux_select_18_tready;

ctrl_dispatcher#(
.D_COUNT(3),
.DISPATCH_WIDTH(2),
.REPLICATED_OUT_NUM(5)
)ctrl_dispatcher(
	 .clk(clk), 
	 .rst(rst) ,
	//selector wire
	.m_selector_tdata({ctrl_demux_select_10_tdata,ctrl_demux_select_12_tdata,ctrl_demux_select_14_tdata,ctrl_demux_select_16_tdata,ctrl_demux_select_18_tdata}),
	.m_selector_tvalid({ctrl_demux_select_10_tvalid,ctrl_demux_select_12_tvalid,ctrl_demux_select_14_tvalid,ctrl_demux_select_16_tvalid,ctrl_demux_select_18_tvalid}),
	.m_selector_tready({ctrl_demux_select_10_tready,ctrl_demux_select_12_tready,ctrl_demux_select_14_tready,ctrl_demux_select_16_tready,ctrl_demux_select_18_tready})
);


endmodule
