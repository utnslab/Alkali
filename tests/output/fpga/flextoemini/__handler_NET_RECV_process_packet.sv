module __handler_NET_RECV_process_packet#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports BUF
	input wire [512-1:0] NET_RECV_1_tdata ,
	input wire [64-1:0] NET_RECV_1_tkeep ,
	input wire  NET_RECV_1_tlast ,
	input wire  NET_RECV_1_tvalid ,
	output wire  NET_RECV_1_tready,
	//output ports STRUCT
	output wire [96-1:0] outport_3_2_tdata ,
	output wire  outport_3_2_tvalid ,
	input wire  outport_3_2_tready
);
//inited_STRUCT
 wire [96-1:0] inited_STRUCT_4_tdata=0;
 wire  inited_STRUCT_4_tvalid=1;
 wire  inited_STRUCT_4_tready;

//extract_module_5 output buf
 wire [512-1:0] bufvar_6_tdata;
 wire [64-1:0] bufvar_6_tkeep;
 wire  bufvar_6_tvalid;
 wire  bufvar_6_tready;
 wire  bufvar_6_tlast;

//extract_module_5 output struct
 wire [160-1:0] structvar_7_tdata;
 wire  structvar_7_tvalid;
 wire  structvar_7_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(160)
)extract_module_5(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({NET_RECV_1_tdata}),
	.s_inbuf_axis_tkeep({NET_RECV_1_tkeep}),
	.s_inbuf_axis_tlast({NET_RECV_1_tlast}),
	.s_inbuf_axis_tvalid({NET_RECV_1_tvalid}),
	.s_inbuf_axis_tready({NET_RECV_1_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_6_tdata}),
	.m_outbuf_axis_tkeep({bufvar_6_tkeep}),
	.m_outbuf_axis_tlast({bufvar_6_tlast}),
	.m_outbuf_axis_tvalid({bufvar_6_tvalid}),
	.m_outbuf_axis_tready({bufvar_6_tready}),
	//output struct
	.m_extracted_axis_tdata({structvar_7_tdata}),
	.m_extracted_axis_tvalid({structvar_7_tvalid}),
	.m_extracted_axis_tready({structvar_7_tready})
);

//const_INT
 wire [32-1:0] const_INT_8_tdata=100;
 wire  const_INT_8_tvalid=1;
 wire  const_INT_8_tready;

//struct_assign_9 output struct
 wire [96-1:0] structvar_10_tdata;
 wire  structvar_10_tvalid;
 wire  structvar_10_tready;

struct_assign#(
.STRUCT_WIDTH(96),
.ASSIGN_OFFSET(32),
.ASSIGN_SIZE(32)
)struct_assign_9(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({inited_STRUCT_4_tdata}),
	.s_struct_axis_tvalid({inited_STRUCT_4_tvalid}),
	.s_struct_axis_tready({inited_STRUCT_4_tready}),
	//input val
	.s_assignv_axis_tdata({const_INT_8_tdata}),
	.s_assignv_axis_tvalid({const_INT_8_tvalid}),
	.s_assignv_axis_tready({const_INT_8_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_10_tdata}),
	.m_struct_axis_tvalid({structvar_10_tvalid}),
	.m_struct_axis_tready({structvar_10_tready})
);

//const_INT
 wire [32-1:0] const_INT_11_tdata=0;
 wire  const_INT_11_tvalid=1;
 wire  const_INT_11_tready;

//struct_assign_12 output struct
 wire [96-1:0] structvar_13_tdata;
 wire  structvar_13_tvalid;
 wire  structvar_13_tready;

struct_assign#(
.STRUCT_WIDTH(96),
.ASSIGN_OFFSET(64),
.ASSIGN_SIZE(32)
)struct_assign_12(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_10_tdata}),
	.s_struct_axis_tvalid({structvar_10_tvalid}),
	.s_struct_axis_tready({structvar_10_tready}),
	//input val
	.s_assignv_axis_tdata({const_INT_11_tdata}),
	.s_assignv_axis_tvalid({const_INT_11_tvalid}),
	.s_assignv_axis_tready({const_INT_11_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_13_tdata}),
	.m_struct_axis_tvalid({structvar_13_tvalid}),
	.m_struct_axis_tready({structvar_13_tready})
);

//Access Struct
 wire [8-1:0] struct_accessed_INT_15_tdata;
 wire  struct_accessed_INT_15_tvalid;
 wire  struct_accessed_INT_15_tready;

//Struct Assign new Struct
 wire [160-1:0] structvar_16_tdata;
 wire  structvar_16_tvalid;
 wire  structvar_16_tready;

struct_access#(
.STRUCT_WIDTH(160),
.ACCESS_OFFSET(104),
.ACCESS_SIZE(8)
)struct_access_14(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_7_tdata}),
	.s_struct_axis_tvalid({structvar_7_tvalid}),
	.s_struct_axis_tready({structvar_7_tready}),
	//struct output
	.m_struct_axis_tdata({structvar_16_tdata}),
	.m_struct_axis_tvalid({structvar_16_tvalid}),
	.m_struct_axis_tready({structvar_16_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_15_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_15_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_15_tready})
);

//struct_assign_17 output struct
 wire [96-1:0] structvar_18_tdata;
 wire  structvar_18_tvalid;
 wire  structvar_18_tready;

struct_assign#(
.STRUCT_WIDTH(96),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(8)
)struct_assign_17(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_13_tdata}),
	.s_struct_axis_tvalid({structvar_13_tvalid}),
	.s_struct_axis_tready({structvar_13_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_15_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_15_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_15_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_18_tdata}),
	.m_struct_axis_tvalid({structvar_18_tvalid}),
	.m_struct_axis_tready({structvar_18_tready})
);

 assign outport_3_2_tdata = structvar_18_tdata;
 assign outport_3_2_tvalid = structvar_18_tvalid;
 assign structvar_18_tready = outport_3_2_tready;


endmodule
