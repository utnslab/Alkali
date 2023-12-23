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
	//output ports BUF
	output wire [512-1:0] outport_1_1_tdata ,
	output wire [64-1:0] outport_1_1_tkeep ,
	output wire  outport_1_1_tlast ,
	output wire  outport_1_1_tvalid ,
	input wire  outport_1_1_tready
);
//inited_BUF
 wire [512-1:0] inited_BUF_2_tdata=0;
 wire [64-1:0] inited_BUF_2_tkeep=0;
 wire  inited_BUF_2_tvalid=1;
 wire  inited_BUF_2_tready;
 wire  inited_BUF_2_tlast=1;

//inited_BUF
 wire [512-1:0] inited_BUF_2_0_tdata;
 wire [64-1:0] inited_BUF_2_0_tkeep;
 wire  inited_BUF_2_0_tvalid;
 wire  inited_BUF_2_0_tready;
 wire  inited_BUF_2_0_tlast;

//inited_BUF
 wire [512-1:0] inited_BUF_2_1_tdata;
 wire [64-1:0] inited_BUF_2_1_tkeep;
 wire  inited_BUF_2_1_tvalid;
 wire  inited_BUF_2_1_tready;
 wire  inited_BUF_2_1_tlast;

axis_replication#(
.DATA_WIDTH(512),
.IF_STREAM(1),
.REAPLICA_COUNT(2)
)axis_replication_3(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({inited_BUF_2_tdata}),
	.s_axis_in_tkeep({inited_BUF_2_tkeep}),
	.s_axis_in_tlast({inited_BUF_2_tlast}),
	.s_axis_in_tvalid({inited_BUF_2_tvalid}),
	.s_axis_in_tready({inited_BUF_2_tready}),
	//
	.m_axis_out_tdata({inited_BUF_2_0_tdata,inited_BUF_2_1_tdata}),
	.m_axis_out_tkeep({inited_BUF_2_0_tkeep,inited_BUF_2_1_tkeep}),
	.m_axis_out_tlast({inited_BUF_2_0_tlast,inited_BUF_2_1_tlast}),
	.m_axis_out_tvalid({inited_BUF_2_0_tvalid,inited_BUF_2_1_tvalid}),
	.m_axis_out_tready({inited_BUF_2_0_tready,inited_BUF_2_1_tready})
);

//extract_module_4 output struct
 wire [112-1:0] structvar_5_tdata;
 wire  structvar_5_tvalid;
 wire  structvar_5_tready;

//extract_module_4 output struct
 wire [112-1:0] structvar_5_0_tdata;
 wire  structvar_5_0_tvalid;
 wire  structvar_5_0_tready;

//extract_module_4 output struct
 wire [112-1:0] structvar_5_1_tdata;
 wire  structvar_5_1_tvalid;
 wire  structvar_5_1_tready;

//extract_module_4 output struct
 wire [112-1:0] structvar_5_2_tdata;
 wire  structvar_5_2_tvalid;
 wire  structvar_5_2_tready;

axis_replication#(
.DATA_WIDTH(112),
.IF_STREAM(0),
.REAPLICA_COUNT(3)
)axis_replication_6(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({structvar_5_tdata}),
	.s_axis_in_tvalid({structvar_5_tvalid}),
	.s_axis_in_tready({structvar_5_tready}),
	//
	.m_axis_out_tdata({structvar_5_0_tdata,structvar_5_1_tdata,structvar_5_2_tdata}),
	.m_axis_out_tvalid({structvar_5_0_tvalid,structvar_5_1_tvalid,structvar_5_2_tvalid}),
	.m_axis_out_tready({structvar_5_0_tready,structvar_5_1_tready,structvar_5_2_tready})
);

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(112)
)extract_module_4(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({NET_RECV_1_tdata}),
	.s_inbuf_axis_tkeep({NET_RECV_1_tkeep}),
	.s_inbuf_axis_tlast({NET_RECV_1_tlast}),
	.s_inbuf_axis_tvalid({NET_RECV_1_tvalid}),
	.s_inbuf_axis_tready({NET_RECV_1_tready}),
	//output struct
	.m_extracted_axis_tdata({structvar_5_tdata}),
	.m_extracted_axis_tvalid({structvar_5_tvalid}),
	.m_extracted_axis_tready({structvar_5_tready})
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_8_tdata;
 wire  struct_accessed_INT_8_tvalid;
 wire  struct_accessed_INT_8_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(48),
.ACCESS_SIZE(48)
)struct_access_7(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_5_0_tdata}),
	.s_struct_axis_tvalid({structvar_5_0_tvalid}),
	.s_struct_axis_tready({structvar_5_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_8_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_8_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_8_tready})
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_10_tdata;
 wire  struct_accessed_INT_10_tvalid;
 wire  struct_accessed_INT_10_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(48)
)struct_access_9(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_5_1_tdata}),
	.s_struct_axis_tvalid({structvar_5_1_tvalid}),
	.s_struct_axis_tready({structvar_5_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_10_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_10_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_10_tready})
);

//struct_assign_11 output struct
 wire [112-1:0] structvar_12_tdata;
 wire  structvar_12_tvalid;
 wire  structvar_12_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(48),
.ASSIGN_SIZE(48)
)struct_assign_11(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_5_2_tdata}),
	.s_struct_axis_tvalid({structvar_5_2_tvalid}),
	.s_struct_axis_tready({structvar_5_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_10_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_10_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_10_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_12_tdata}),
	.m_struct_axis_tvalid({structvar_12_tvalid}),
	.m_struct_axis_tready({structvar_12_tready})
);

//struct_assign_13 output struct
 wire [112-1:0] structvar_14_tdata;
 wire  structvar_14_tvalid;
 wire  structvar_14_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(48)
)struct_assign_13(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_12_tdata}),
	.s_struct_axis_tvalid({structvar_12_tvalid}),
	.s_struct_axis_tready({structvar_12_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_8_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_8_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_8_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_14_tdata}),
	.m_struct_axis_tvalid({structvar_14_tvalid}),
	.m_struct_axis_tready({structvar_14_tready})
);

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(112)
)emit_module_15(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({inited_BUF_2_0_tdata}),
	.s_inbuf_axis_tkeep({inited_BUF_2_0_tkeep}),
	.s_inbuf_axis_tlast({inited_BUF_2_0_tlast}),
	.s_inbuf_axis_tvalid({inited_BUF_2_0_tvalid}),
	.s_inbuf_axis_tready({inited_BUF_2_0_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_14_tdata}),
	.s_struct_axis_tvalid({structvar_14_tvalid}),
	.s_struct_axis_tready({structvar_14_tready})
);

 assign outport_1_1_tdata = inited_BUF_2_1_tdata;
 assign outport_1_1_tvalid = inited_BUF_2_1_tvalid;
 assign inited_BUF_2_1_tready = outport_1_1_tready;
 assign outport_1_1_tkeep = inited_BUF_2_1_tkeep;
 assign outport_1_1_tlast = inited_BUF_2_1_tlast;


endmodule
