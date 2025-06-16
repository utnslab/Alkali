module __handler_NET_RECV_process_packet#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports BUF
	input wire [512-1:0] NET_RECV_0_tdata ,
	input wire [64-1:0] NET_RECV_0_tkeep ,
	input wire  NET_RECV_0_tlast ,
	input wire  NET_RECV_0_tvalid ,
	output wire  NET_RECV_0_tready,
	//output ports STRUCT
	output wire [96-1:0] outport_0_1_tdata ,
	output wire  outport_0_1_tvalid ,
	input wire  outport_0_1_tready,
	//output ports BUF
	output wire [512-1:0] outport_0_2_tdata ,
	output wire [64-1:0] outport_0_2_tkeep ,
	output wire  outport_0_2_tlast ,
	output wire  outport_0_2_tvalid ,
	input wire  outport_0_2_tready,
	//output ports STRUCT
	output wire [184-1:0] outport_0_3_tdata ,
	output wire  outport_0_3_tvalid ,
	input wire  outport_0_3_tready,
	//output ports STRUCT
	output wire [112-1:0] outport_0_4_tdata ,
	output wire  outport_0_4_tvalid ,
	input wire  outport_0_4_tready,
	//output ports STRUCT
	output wire [160-1:0] outport_0_5_tdata ,
	output wire  outport_0_5_tvalid ,
	input wire  outport_0_5_tready
);
//const_INT
 wire [32-1:0] const_INT_1_tdata=0;
 wire  const_INT_1_tvalid=1;
 wire  const_INT_1_tready;

//const_INT
 wire [64-1:0] const_INT_2_tdata=14;
 wire  const_INT_2_tvalid=1;
 wire  const_INT_2_tready;

//inited_STRUCT
 wire [96-1:0] inited_STRUCT_3_tdata=0;
 wire  inited_STRUCT_3_tvalid=1;
 wire  inited_STRUCT_3_tready;

//extract_module_4 output buf
 wire [512-1:0] bufvar_5_tdata;
 wire [64-1:0] bufvar_5_tkeep;
 wire  bufvar_5_tvalid;
 wire  bufvar_5_tready;
 wire  bufvar_5_tlast;

//extract_module_4 output struct
 wire [112-1:0] structvar_6_tdata;
 wire  structvar_6_tvalid;
 wire  structvar_6_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(112)
)extract_module_4(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({NET_RECV_0_tdata}),
	.s_inbuf_axis_tkeep({NET_RECV_0_tkeep}),
	.s_inbuf_axis_tlast({NET_RECV_0_tlast}),
	.s_inbuf_axis_tvalid({NET_RECV_0_tvalid}),
	.s_inbuf_axis_tready({NET_RECV_0_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_5_tdata}),
	.m_outbuf_axis_tkeep({bufvar_5_tkeep}),
	.m_outbuf_axis_tlast({bufvar_5_tlast}),
	.m_outbuf_axis_tvalid({bufvar_5_tvalid}),
	.m_outbuf_axis_tready({bufvar_5_tready}),
	//output struct
	.m_extracted_axis_tdata({structvar_6_tdata}),
	.m_extracted_axis_tvalid({structvar_6_tvalid}),
	.m_extracted_axis_tready({structvar_6_tready})
);

//extract_module_7 output buf
 wire [512-1:0] bufvar_8_tdata;
 wire [64-1:0] bufvar_8_tkeep;
 wire  bufvar_8_tvalid;
 wire  bufvar_8_tready;
 wire  bufvar_8_tlast;

//extract_module_7 output struct
 wire [184-1:0] structvar_9_tdata;
 wire  structvar_9_tvalid;
 wire  structvar_9_tready;

//extract_module_7 output struct
 wire [184-1:0] structvar_9_0_tdata;
 wire  structvar_9_0_tvalid;
 wire  structvar_9_0_tready;

//extract_module_7 output struct
 wire [184-1:0] structvar_9_1_tdata;
 wire  structvar_9_1_tvalid;
 wire  structvar_9_1_tready;

axis_replication#(
.DATA_WIDTH(184),
.IF_STREAM(0),
.REAPLICA_COUNT(2)
)axis_replication_10(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({structvar_9_tdata}),
	.s_axis_in_tvalid({structvar_9_tvalid}),
	.s_axis_in_tready({structvar_9_tready}),
	//
	.m_axis_out_tdata({structvar_9_0_tdata,structvar_9_1_tdata}),
	.m_axis_out_tvalid({structvar_9_0_tvalid,structvar_9_1_tvalid}),
	.m_axis_out_tready({structvar_9_0_tready,structvar_9_1_tready})
);

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(184)
)extract_module_7(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_5_tdata}),
	.s_inbuf_axis_tkeep({bufvar_5_tkeep}),
	.s_inbuf_axis_tlast({bufvar_5_tlast}),
	.s_inbuf_axis_tvalid({bufvar_5_tvalid}),
	.s_inbuf_axis_tready({bufvar_5_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_8_tdata}),
	.m_outbuf_axis_tkeep({bufvar_8_tkeep}),
	.m_outbuf_axis_tlast({bufvar_8_tlast}),
	.m_outbuf_axis_tvalid({bufvar_8_tvalid}),
	.m_outbuf_axis_tready({bufvar_8_tready}),
	//output struct
	.m_extracted_axis_tdata({structvar_9_tdata}),
	.m_extracted_axis_tvalid({structvar_9_tvalid}),
	.m_extracted_axis_tready({structvar_9_tready})
);

//extract_module_11 output buf
 wire [512-1:0] bufvar_12_tdata;
 wire [64-1:0] bufvar_12_tkeep;
 wire  bufvar_12_tvalid;
 wire  bufvar_12_tready;
 wire  bufvar_12_tlast;

//extract_module_11 output struct
 wire [160-1:0] structvar_13_tdata;
 wire  structvar_13_tvalid;
 wire  structvar_13_tready;

//extract_module_11 output struct
 wire [160-1:0] structvar_13_0_tdata;
 wire  structvar_13_0_tvalid;
 wire  structvar_13_0_tready;

//extract_module_11 output struct
 wire [160-1:0] structvar_13_1_tdata;
 wire  structvar_13_1_tvalid;
 wire  structvar_13_1_tready;

axis_replication#(
.DATA_WIDTH(160),
.IF_STREAM(0),
.REAPLICA_COUNT(2)
)axis_replication_14(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({structvar_13_tdata}),
	.s_axis_in_tvalid({structvar_13_tvalid}),
	.s_axis_in_tready({structvar_13_tready}),
	//
	.m_axis_out_tdata({structvar_13_0_tdata,structvar_13_1_tdata}),
	.m_axis_out_tvalid({structvar_13_0_tvalid,structvar_13_1_tvalid}),
	.m_axis_out_tready({structvar_13_0_tready,structvar_13_1_tready})
);

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(160)
)extract_module_11(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_8_tdata}),
	.s_inbuf_axis_tkeep({bufvar_8_tkeep}),
	.s_inbuf_axis_tlast({bufvar_8_tlast}),
	.s_inbuf_axis_tvalid({bufvar_8_tvalid}),
	.s_inbuf_axis_tready({bufvar_8_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_12_tdata}),
	.m_outbuf_axis_tkeep({bufvar_12_tkeep}),
	.m_outbuf_axis_tlast({bufvar_12_tlast}),
	.m_outbuf_axis_tvalid({bufvar_12_tvalid}),
	.m_outbuf_axis_tready({bufvar_12_tready}),
	//output struct
	.m_extracted_axis_tdata({structvar_13_tdata}),
	.m_extracted_axis_tvalid({structvar_13_tvalid}),
	.m_extracted_axis_tready({structvar_13_tready})
);

//Access Struct
 wire [16-1:0] struct_accessed_INT_16_tdata;
 wire  struct_accessed_INT_16_tvalid;
 wire  struct_accessed_INT_16_tready;

struct_access#(
.STRUCT_WIDTH(184),
.ACCESS_OFFSET(8),
.ACCESS_SIZE(16)
)struct_access_15(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_9_0_tdata}),
	.s_struct_axis_tvalid({structvar_9_0_tvalid}),
	.s_struct_axis_tready({structvar_9_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_16_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_16_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_16_tready})
);

//Arithmetic OP Out
 wire [16-1:0] ADD_17_out_INT_18_tdata;
 wire  ADD_17_out_INT_18_tvalid;
 wire  ADD_17_out_INT_18_tready;

ALU#(
.LVAL_SIZE(16),
.RVAL_SIZE(64),
.RESULT_SIZE(16),
.OPID(1)
)ADD_17(
	 .clk(clk), 
	 .rst(rst) ,
	//lval input
	.s_lval_axis_tdata({struct_accessed_INT_16_tdata}),
	.s_lval_axis_tvalid({struct_accessed_INT_16_tvalid}),
	.s_lval_axis_tready({struct_accessed_INT_16_tready}),
	//rval input
	.s_rval_axis_tdata({const_INT_2_tdata}),
	.s_rval_axis_tvalid({const_INT_2_tvalid}),
	.s_rval_axis_tready({const_INT_2_tready}),
	//output val
	.m_val_axis_tdata({ADD_17_out_INT_18_tdata}),
	.m_val_axis_tvalid({ADD_17_out_INT_18_tvalid}),
	.m_val_axis_tready({ADD_17_out_INT_18_tready})
);

//bitcast dst
 wire [32-1:0] bitcasted_19_tdata;
 wire  bitcasted_19_tvalid;
 wire  bitcasted_19_tready;

 assign bitcasted_19_tdata = ADD_17_out_INT_18_tdata;
 assign bitcasted_19_tvalid = ADD_17_out_INT_18_tvalid;
 assign ADD_17_out_INT_18_tready = bitcasted_19_tready;

//struct_assign_20 output struct
 wire [96-1:0] structvar_21_tdata;
 wire  structvar_21_tvalid;
 wire  structvar_21_tready;

struct_assign#(
.STRUCT_WIDTH(96),
.ASSIGN_OFFSET(32),
.ASSIGN_SIZE(32)
)struct_assign_20(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({inited_STRUCT_3_tdata}),
	.s_struct_axis_tvalid({inited_STRUCT_3_tvalid}),
	.s_struct_axis_tready({inited_STRUCT_3_tready}),
	//input val
	.s_assignv_axis_tdata({bitcasted_19_tdata}),
	.s_assignv_axis_tvalid({bitcasted_19_tvalid}),
	.s_assignv_axis_tready({bitcasted_19_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_21_tdata}),
	.m_struct_axis_tvalid({structvar_21_tvalid}),
	.m_struct_axis_tready({structvar_21_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_23_tdata;
 wire  struct_accessed_INT_23_tvalid;
 wire  struct_accessed_INT_23_tready;

struct_access#(
.STRUCT_WIDTH(160),
.ACCESS_OFFSET(32),
.ACCESS_SIZE(32)
)struct_access_22(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_13_0_tdata}),
	.s_struct_axis_tvalid({structvar_13_0_tvalid}),
	.s_struct_axis_tready({structvar_13_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_23_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_23_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_23_tready})
);

//struct_assign_24 output struct
 wire [96-1:0] structvar_25_tdata;
 wire  structvar_25_tvalid;
 wire  structvar_25_tready;

struct_assign#(
.STRUCT_WIDTH(96),
.ASSIGN_OFFSET(64),
.ASSIGN_SIZE(32)
)struct_assign_24(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_21_tdata}),
	.s_struct_axis_tvalid({structvar_21_tvalid}),
	.s_struct_axis_tready({structvar_21_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_23_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_23_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_23_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_25_tdata}),
	.m_struct_axis_tvalid({structvar_25_tvalid}),
	.m_struct_axis_tready({structvar_25_tready})
);

//struct_assign_26 output struct
 wire [96-1:0] structvar_27_tdata;
 wire  structvar_27_tvalid;
 wire  structvar_27_tready;

struct_assign#(
.STRUCT_WIDTH(96),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(32)
)struct_assign_26(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_25_tdata}),
	.s_struct_axis_tvalid({structvar_25_tvalid}),
	.s_struct_axis_tready({structvar_25_tready}),
	//input val
	.s_assignv_axis_tdata({const_INT_1_tdata}),
	.s_assignv_axis_tvalid({const_INT_1_tvalid}),
	.s_assignv_axis_tready({const_INT_1_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_27_tdata}),
	.m_struct_axis_tvalid({structvar_27_tvalid}),
	.m_struct_axis_tready({structvar_27_tready})
);

 assign outport_0_1_tdata = structvar_27_tdata;
 assign outport_0_1_tvalid = structvar_27_tvalid;
 assign structvar_27_tready = outport_0_1_tready;

 assign outport_0_2_tdata = bufvar_12_tdata;
 assign outport_0_2_tvalid = bufvar_12_tvalid;
 assign bufvar_12_tready = outport_0_2_tready;
 assign outport_0_2_tkeep = bufvar_12_tkeep;
 assign outport_0_2_tlast = bufvar_12_tlast;

 assign outport_0_3_tdata = structvar_9_1_tdata;
 assign outport_0_3_tvalid = structvar_9_1_tvalid;
 assign structvar_9_1_tready = outport_0_3_tready;

 assign outport_0_4_tdata = structvar_6_tdata;
 assign outport_0_4_tvalid = structvar_6_tvalid;
 assign structvar_6_tready = outport_0_4_tready;

 assign outport_0_5_tdata = structvar_13_1_tdata;
 assign outport_0_5_tvalid = structvar_13_1_tvalid;
 assign structvar_13_1_tready = outport_0_5_tready;


endmodule
