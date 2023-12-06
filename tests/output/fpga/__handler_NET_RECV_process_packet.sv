module __handler_NET_RECV_process_packet#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports BUF
	input wire [512-1:0] arg1_tdata ,
	input wire [64-1:0] arg1_tkeep ,
	input wire  arg1_tlast ,
	input wire  arg1_tvalid ,
	output wire  arg1_tready,
	//output ports BUF
	output wire [512-1:0] outport_0_1_tdata ,
	output wire [64-1:0] outport_0_1_tkeep ,
	output wire  outport_0_1_tlast ,
	output wire  outport_0_1_tvalid ,
	input wire  outport_0_1_tready
);
//inited_STRUCT
 wire [112-1:0] inited_STRUCT_1_tdata=0;
 wire  inited_STRUCT_1_tvalid=1;
 wire  inited_STRUCT_1_tready;

//inited_INT
 wire [48-1:0] inited_INT_2_tdata=0;
 wire  inited_INT_2_tvalid=1;
 wire  inited_INT_2_tready;

//inited_STRUCT
 wire [184-1:0] inited_STRUCT_3_tdata=0;
 wire  inited_STRUCT_3_tvalid=1;
 wire  inited_STRUCT_3_tready;

//inited_BUF
 wire [512-1:0] inited_BUF_4_tdata=0;
 wire [64-1:0] inited_BUF_4_tkeep=0;
 wire  inited_BUF_4_tvalid=1;
 wire  inited_BUF_4_tready;
 wire  inited_BUF_4_tlast=1;

//extract_module_5 output buf
 wire [512-1:0] bufvar_6_tdata;
 wire [64-1:0] bufvar_6_tkeep;
 wire  bufvar_6_tvalid;
 wire  bufvar_6_tready;
 wire  bufvar_6_tlast;

//extract_module_5 output struct
 wire [112-1:0] structvar_7_tdata;
 wire  structvar_7_tvalid;
 wire  structvar_7_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(112)
)extract_module_5(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata(arg1_tdata),
	.s_inbuf_axis_tkeep(arg1_tkeep),
	.s_inbuf_axis_tlast(arg1_tlast),
	.s_inbuf_axis_tvalid(arg1_tvalid),
	.s_inbuf_axis_tready(arg1_tready),
	//output buf
	.m_outbuf_axis_tdata(bufvar_6_tdata),
	.m_outbuf_axis_tkeep(bufvar_6_tkeep),
	.m_outbuf_axis_tlast(bufvar_6_tlast),
	.m_outbuf_axis_tvalid(bufvar_6_tvalid),
	.m_outbuf_axis_tready(bufvar_6_tready),
	//output struct
	.m_extracted_axis_tdata(structvar_7_tdata),
	.m_extracted_axis_tvalid(structvar_7_tvalid),
	.m_extracted_axis_tready(structvar_7_tready)
);

//extract_module_8 output buf
 wire [512-1:0] bufvar_9_tdata;
 wire [64-1:0] bufvar_9_tkeep;
 wire  bufvar_9_tvalid;
 wire  bufvar_9_tready;
 wire  bufvar_9_tlast;

//extract_module_8 output struct
 wire [184-1:0] structvar_10_tdata;
 wire  structvar_10_tvalid;
 wire  structvar_10_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(184)
)extract_module_8(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata(bufvar_6_tdata),
	.s_inbuf_axis_tkeep(bufvar_6_tkeep),
	.s_inbuf_axis_tlast(bufvar_6_tlast),
	.s_inbuf_axis_tvalid(bufvar_6_tvalid),
	.s_inbuf_axis_tready(bufvar_6_tready),
	//output buf
	.m_outbuf_axis_tdata(bufvar_9_tdata),
	.m_outbuf_axis_tkeep(bufvar_9_tkeep),
	.m_outbuf_axis_tlast(bufvar_9_tlast),
	.m_outbuf_axis_tvalid(bufvar_9_tvalid),
	.m_outbuf_axis_tready(bufvar_9_tready),
	//output struct
	.m_extracted_axis_tdata(structvar_10_tdata),
	.m_extracted_axis_tvalid(structvar_10_tvalid),
	.m_extracted_axis_tready(structvar_10_tready)
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_12_tdata;
 wire  struct_accessed_INT_12_tvalid;
 wire  struct_accessed_INT_12_tready;

//Struct Assign new Struct
 wire [112-1:0] structvar_13_tdata;
 wire  structvar_13_tvalid;
 wire  structvar_13_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(48),
.ACCESS_SIZE(48)
)struct_access_11(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata(structvar_7_tdata),
	.s_struct_axis_tvalid(structvar_7_tvalid),
	.s_struct_axis_tready(structvar_7_tready),
	//struct output
	.m_struct_axis_tdata(structvar_13_tdata),
	.m_struct_axis_tvalid(structvar_13_tvalid),
	.m_struct_axis_tready(structvar_13_tready),
	//output val
	.m_val_axis_tdata(struct_accessed_INT_12_tdata),
	.m_val_axis_tvalid(struct_accessed_INT_12_tvalid),
	.m_val_axis_tready(struct_accessed_INT_12_tready)
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_15_tdata;
 wire  struct_accessed_INT_15_tvalid;
 wire  struct_accessed_INT_15_tready;

//Struct Assign new Struct
 wire [112-1:0] structvar_16_tdata;
 wire  structvar_16_tvalid;
 wire  structvar_16_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(48)
)struct_access_14(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata(structvar_13_tdata),
	.s_struct_axis_tvalid(structvar_13_tvalid),
	.s_struct_axis_tready(structvar_13_tready),
	//struct output
	.m_struct_axis_tdata(structvar_16_tdata),
	.m_struct_axis_tvalid(structvar_16_tvalid),
	.m_struct_axis_tready(structvar_16_tready),
	//output val
	.m_val_axis_tdata(struct_accessed_INT_15_tdata),
	.m_val_axis_tvalid(struct_accessed_INT_15_tvalid),
	.m_val_axis_tready(struct_accessed_INT_15_tready)
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_18_tdata;
 wire  struct_accessed_INT_18_tvalid;
 wire  struct_accessed_INT_18_tready=1;

//Struct Assign new Struct
 wire [112-1:0] structvar_19_tdata;
 wire  structvar_19_tvalid;
 wire  structvar_19_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(48),
.ACCESS_SIZE(48)
)struct_access_17(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata(structvar_16_tdata),
	.s_struct_axis_tvalid(structvar_16_tvalid),
	.s_struct_axis_tready(structvar_16_tready),
	//struct output
	.m_struct_axis_tdata(structvar_19_tdata),
	.m_struct_axis_tvalid(structvar_19_tvalid),
	.m_struct_axis_tready(structvar_19_tready),
	//output val
	.m_val_axis_tdata(struct_accessed_INT_18_tdata),
	.m_val_axis_tvalid(struct_accessed_INT_18_tvalid),
	.m_val_axis_tready(struct_accessed_INT_18_tready)
);

//struct_assign_20 output struct
 wire [112-1:0] structvar_21_tdata;
 wire  structvar_21_tvalid;
 wire  structvar_21_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(48),
.ASSIGN_SIZE(48)
)struct_assign_20(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata(structvar_19_tdata),
	.s_struct_axis_tvalid(structvar_19_tvalid),
	.s_struct_axis_tready(structvar_19_tready),
	//input val
	.s_assignv_axis_tdata(struct_accessed_INT_15_tdata),
	.s_assignv_axis_tvalid(struct_accessed_INT_15_tvalid),
	.s_assignv_axis_tready(struct_accessed_INT_15_tready),
	//output struct
	.m_struct_axis_tdata(structvar_21_tdata),
	.m_struct_axis_tvalid(structvar_21_tvalid),
	.m_struct_axis_tready(structvar_21_tready)
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_23_tdata;
 wire  struct_accessed_INT_23_tvalid;
 wire  struct_accessed_INT_23_tready=1;

//Struct Assign new Struct
 wire [112-1:0] structvar_24_tdata;
 wire  structvar_24_tvalid;
 wire  structvar_24_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(48)
)struct_access_22(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata(structvar_21_tdata),
	.s_struct_axis_tvalid(structvar_21_tvalid),
	.s_struct_axis_tready(structvar_21_tready),
	//struct output
	.m_struct_axis_tdata(structvar_24_tdata),
	.m_struct_axis_tvalid(structvar_24_tvalid),
	.m_struct_axis_tready(structvar_24_tready),
	//output val
	.m_val_axis_tdata(struct_accessed_INT_23_tdata),
	.m_val_axis_tvalid(struct_accessed_INT_23_tvalid),
	.m_val_axis_tready(struct_accessed_INT_23_tready)
);

//struct_assign_25 output struct
 wire [112-1:0] structvar_26_tdata;
 wire  structvar_26_tvalid;
 wire  structvar_26_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(48)
)struct_assign_25(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata(structvar_24_tdata),
	.s_struct_axis_tvalid(structvar_24_tvalid),
	.s_struct_axis_tready(structvar_24_tready),
	//input val
	.s_assignv_axis_tdata(struct_accessed_INT_12_tdata),
	.s_assignv_axis_tvalid(struct_accessed_INT_12_tvalid),
	.s_assignv_axis_tready(struct_accessed_INT_12_tready),
	//output struct
	.m_struct_axis_tdata(structvar_26_tdata),
	.m_struct_axis_tvalid(structvar_26_tvalid),
	.m_struct_axis_tready(structvar_26_tready)
);

//emit_module_27 output buf
 wire [512-1:0] bufvar_28_tdata;
 wire [64-1:0] bufvar_28_tkeep;
 wire  bufvar_28_tvalid;
 wire  bufvar_28_tready;
 wire  bufvar_28_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(112)
)emit_module_27(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata(inited_BUF_4_tdata),
	.s_inbuf_axis_tkeep(inited_BUF_4_tkeep),
	.s_inbuf_axis_tlast(inited_BUF_4_tlast),
	.s_inbuf_axis_tvalid(inited_BUF_4_tvalid),
	.s_inbuf_axis_tready(inited_BUF_4_tready),
	//input struct/buf
	.s_struct_axis_tdata(structvar_26_tdata),
	.s_struct_axis_tvalid(structvar_26_tvalid),
	.s_struct_axis_tready(structvar_26_tready),
	//output buf
	.m_outbuf_axis_tdata(bufvar_28_tdata),
	.m_outbuf_axis_tkeep(bufvar_28_tkeep),
	.m_outbuf_axis_tlast(bufvar_28_tlast),
	.m_outbuf_axis_tvalid(bufvar_28_tvalid),
	.m_outbuf_axis_tready(bufvar_28_tready)
);

//emit_module_29 output buf
 wire [512-1:0] bufvar_30_tdata;
 wire [64-1:0] bufvar_30_tkeep;
 wire  bufvar_30_tvalid;
 wire  bufvar_30_tready;
 wire  bufvar_30_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(184)
)emit_module_29(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata(bufvar_28_tdata),
	.s_inbuf_axis_tkeep(bufvar_28_tkeep),
	.s_inbuf_axis_tlast(bufvar_28_tlast),
	.s_inbuf_axis_tvalid(bufvar_28_tvalid),
	.s_inbuf_axis_tready(bufvar_28_tready),
	//input struct/buf
	.s_struct_axis_tdata(structvar_10_tdata),
	.s_struct_axis_tvalid(structvar_10_tvalid),
	.s_struct_axis_tready(structvar_10_tready),
	//output buf
	.m_outbuf_axis_tdata(bufvar_30_tdata),
	.m_outbuf_axis_tkeep(bufvar_30_tkeep),
	.m_outbuf_axis_tlast(bufvar_30_tlast),
	.m_outbuf_axis_tvalid(bufvar_30_tvalid),
	.m_outbuf_axis_tready(bufvar_30_tready)
);

//emit_module_31 output buf
 wire [512-1:0] bufvar_32_tdata;
 wire [64-1:0] bufvar_32_tkeep;
 wire  bufvar_32_tvalid;
 wire  bufvar_32_tready;
 wire  bufvar_32_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(1),
.INPUT_BUF_STRUCT_WIDTH(512)
)emit_module_31(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata(bufvar_30_tdata),
	.s_inbuf_axis_tkeep(bufvar_30_tkeep),
	.s_inbuf_axis_tlast(bufvar_30_tlast),
	.s_inbuf_axis_tvalid(bufvar_30_tvalid),
	.s_inbuf_axis_tready(bufvar_30_tready),
	//input struct/buf
	.s_struct_axis_tdata(bufvar_9_tdata),
	.s_struct_axis_tkeep(bufvar_9_tkeep),
	.s_struct_axis_tlast(bufvar_9_tlast),
	.s_struct_axis_tvalid(bufvar_9_tvalid),
	.s_struct_axis_tready(bufvar_9_tready),
	//output buf
	.m_outbuf_axis_tdata(bufvar_32_tdata),
	.m_outbuf_axis_tkeep(bufvar_32_tkeep),
	.m_outbuf_axis_tlast(bufvar_32_tlast),
	.m_outbuf_axis_tvalid(bufvar_32_tvalid),
	.m_outbuf_axis_tready(bufvar_32_tready)
);

 assign outport_0_1_tdata = bufvar_32_tdata;
 assign outport_0_1_tvalid = bufvar_32_tvalid;
 assign bufvar_32_tready = outport_0_1_tready;
 assign outport_0_1_tkeep = bufvar_32_tkeep;
 assign outport_0_1_tlast = bufvar_32_tlast;


endmodule
