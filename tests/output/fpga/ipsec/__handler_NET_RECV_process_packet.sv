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
	input wire  DECRYPT_REQ_4_tready
);
//inited_STRUCT
 wire [272-1:0] inited_STRUCT_0_tdata=0;
 wire  inited_STRUCT_0_tvalid=1;
 wire  inited_STRUCT_0_tready;

//extract_module_1 output buf
 wire [512-1:0] bufvar_2_tdata;
 wire [64-1:0] bufvar_2_tkeep;
 wire  bufvar_2_tvalid;
 wire  bufvar_2_tready;
 wire  bufvar_2_tlast;

//extract_module_1 output struct
 wire [112-1:0] structvar_3_tdata;
 wire  structvar_3_tvalid;
 wire  structvar_3_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(112)
)extract_module_1(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({NET_RECV_0_tdata}),
	.s_inbuf_axis_tkeep({NET_RECV_0_tkeep}),
	.s_inbuf_axis_tlast({NET_RECV_0_tlast}),
	.s_inbuf_axis_tvalid({NET_RECV_0_tvalid}),
	.s_inbuf_axis_tready({NET_RECV_0_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_2_tdata}),
	.m_outbuf_axis_tkeep({bufvar_2_tkeep}),
	.m_outbuf_axis_tlast({bufvar_2_tlast}),
	.m_outbuf_axis_tvalid({bufvar_2_tvalid}),
	.m_outbuf_axis_tready({bufvar_2_tready}),
	//output struct
	.m_extracted_axis_tdata({structvar_3_tdata}),
	.m_extracted_axis_tvalid({structvar_3_tvalid}),
	.m_extracted_axis_tready({structvar_3_tready})
);

//extract_module_4 output buf
 wire [512-1:0] bufvar_5_tdata;
 wire [64-1:0] bufvar_5_tkeep;
 wire  bufvar_5_tvalid;
 wire  bufvar_5_tready;
 wire  bufvar_5_tlast;

//extract_module_4 output struct
 wire [184-1:0] structvar_6_tdata;
 wire  structvar_6_tvalid;
 wire  structvar_6_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(184)
)extract_module_4(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_2_tdata}),
	.s_inbuf_axis_tkeep({bufvar_2_tkeep}),
	.s_inbuf_axis_tlast({bufvar_2_tlast}),
	.s_inbuf_axis_tvalid({bufvar_2_tvalid}),
	.s_inbuf_axis_tready({bufvar_2_tready}),
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
 wire [64-1:0] structvar_9_tdata;
 wire  structvar_9_tvalid;
 wire  structvar_9_tready;

extract#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.EXTRACTED_STRUCT_WIDTH(64)
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

 assign DECRYPT_REQ_0_tdata = bufvar_8_tdata;
 assign DECRYPT_REQ_0_tvalid = bufvar_8_tvalid;
 assign bufvar_8_tready = DECRYPT_REQ_0_tready;
 assign DECRYPT_REQ_0_tkeep = bufvar_8_tkeep;
 assign DECRYPT_REQ_0_tlast = bufvar_8_tlast;

 assign DECRYPT_REQ_1_tdata = inited_STRUCT_0_tdata;
 assign DECRYPT_REQ_1_tvalid = inited_STRUCT_0_tvalid;
 assign inited_STRUCT_0_tready = DECRYPT_REQ_1_tready;

 assign DECRYPT_REQ_2_tdata = structvar_6_tdata;
 assign DECRYPT_REQ_2_tvalid = structvar_6_tvalid;
 assign structvar_6_tready = DECRYPT_REQ_2_tready;

 assign DECRYPT_REQ_3_tdata = structvar_3_tdata;
 assign DECRYPT_REQ_3_tvalid = structvar_3_tvalid;
 assign structvar_3_tready = DECRYPT_REQ_3_tready;

 assign DECRYPT_REQ_4_tdata = structvar_9_tdata;
 assign DECRYPT_REQ_4_tvalid = structvar_9_tvalid;
 assign structvar_9_tready = DECRYPT_REQ_4_tready;


endmodule
