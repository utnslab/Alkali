module __handler_ACK_GEN_ack_gen#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports STRUCT
	input wire [96-1:0] arg_151_tdata ,
	input wire  arg_151_tvalid ,
	output wire  arg_151_tready,
	//input ports BUF
	input wire [512-1:0] arg_152_tdata ,
	input wire [64-1:0] arg_152_tkeep ,
	input wire  arg_152_tlast ,
	input wire  arg_152_tvalid ,
	output wire  arg_152_tready,
	//input ports STRUCT
	input wire [184-1:0] arg_153_tdata ,
	input wire  arg_153_tvalid ,
	output wire  arg_153_tready,
	//input ports STRUCT
	input wire [112-1:0] arg_154_tdata ,
	input wire  arg_154_tvalid ,
	output wire  arg_154_tready,
	//input ports STRUCT
	input wire [160-1:0] arg_155_tdata ,
	input wire  arg_155_tvalid ,
	output wire  arg_155_tready,
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
//
 wire [96-1:0] arg_151_0_tdata;
 wire  arg_151_0_tvalid;
 wire  arg_151_0_tready;

//
 wire [96-1:0] arg_151_1_tdata;
 wire  arg_151_1_tvalid;
 wire  arg_151_1_tready;

axis_replication#(
.DATA_WIDTH(96),
.IF_STREAM(0),
.REAPLICA_COUNT(2)
)axis_replication_156(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_151_tdata}),
	.s_axis_in_tvalid({arg_151_tvalid}),
	.s_axis_in_tready({arg_151_tready}),
	//
	.m_axis_out_tdata({arg_151_0_tdata,arg_151_1_tdata}),
	.m_axis_out_tvalid({arg_151_0_tvalid,arg_151_1_tvalid}),
	.m_axis_out_tready({arg_151_0_tready,arg_151_1_tready})
);

//
 wire [512-1:0] arg_152_0_tdata;
 wire [64-1:0] arg_152_0_tkeep;
 wire  arg_152_0_tvalid;
 wire  arg_152_0_tready;
 wire  arg_152_0_tlast;

//
 wire [512-1:0] arg_152_1_tdata;
 wire [64-1:0] arg_152_1_tkeep;
 wire  arg_152_1_tvalid;
 wire  arg_152_1_tready;
 wire  arg_152_1_tlast;

axis_replication#(
.DATA_WIDTH(512),
.IF_STREAM(1),
.REAPLICA_COUNT(2)
)axis_replication_157(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_152_tdata}),
	.s_axis_in_tkeep({arg_152_tkeep}),
	.s_axis_in_tlast({arg_152_tlast}),
	.s_axis_in_tvalid({arg_152_tvalid}),
	.s_axis_in_tready({arg_152_tready}),
	//
	.m_axis_out_tdata({arg_152_0_tdata,arg_152_1_tdata}),
	.m_axis_out_tkeep({arg_152_0_tkeep,arg_152_1_tkeep}),
	.m_axis_out_tlast({arg_152_0_tlast,arg_152_1_tlast}),
	.m_axis_out_tvalid({arg_152_0_tvalid,arg_152_1_tvalid}),
	.m_axis_out_tready({arg_152_0_tready,arg_152_1_tready})
);

//
 wire [184-1:0] arg_153_0_tdata;
 wire  arg_153_0_tvalid;
 wire  arg_153_0_tready;

//
 wire [184-1:0] arg_153_1_tdata;
 wire  arg_153_1_tvalid;
 wire  arg_153_1_tready;

//
 wire [184-1:0] arg_153_2_tdata;
 wire  arg_153_2_tvalid;
 wire  arg_153_2_tready;

//
 wire [184-1:0] arg_153_3_tdata;
 wire  arg_153_3_tvalid;
 wire  arg_153_3_tready;

axis_replication#(
.DATA_WIDTH(184),
.IF_STREAM(0),
.REAPLICA_COUNT(4)
)axis_replication_158(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_153_tdata}),
	.s_axis_in_tvalid({arg_153_tvalid}),
	.s_axis_in_tready({arg_153_tready}),
	//
	.m_axis_out_tdata({arg_153_0_tdata,arg_153_1_tdata,arg_153_2_tdata,arg_153_3_tdata}),
	.m_axis_out_tvalid({arg_153_0_tvalid,arg_153_1_tvalid,arg_153_2_tvalid,arg_153_3_tvalid}),
	.m_axis_out_tready({arg_153_0_tready,arg_153_1_tready,arg_153_2_tready,arg_153_3_tready})
);

//
 wire [112-1:0] arg_154_0_tdata;
 wire  arg_154_0_tvalid;
 wire  arg_154_0_tready;

//
 wire [112-1:0] arg_154_1_tdata;
 wire  arg_154_1_tvalid;
 wire  arg_154_1_tready;

//
 wire [112-1:0] arg_154_2_tdata;
 wire  arg_154_2_tvalid;
 wire  arg_154_2_tready;

//
 wire [112-1:0] arg_154_3_tdata;
 wire  arg_154_3_tvalid;
 wire  arg_154_3_tready;

axis_replication#(
.DATA_WIDTH(112),
.IF_STREAM(0),
.REAPLICA_COUNT(4)
)axis_replication_159(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_154_tdata}),
	.s_axis_in_tvalid({arg_154_tvalid}),
	.s_axis_in_tready({arg_154_tready}),
	//
	.m_axis_out_tdata({arg_154_0_tdata,arg_154_1_tdata,arg_154_2_tdata,arg_154_3_tdata}),
	.m_axis_out_tvalid({arg_154_0_tvalid,arg_154_1_tvalid,arg_154_2_tvalid,arg_154_3_tvalid}),
	.m_axis_out_tready({arg_154_0_tready,arg_154_1_tready,arg_154_2_tready,arg_154_3_tready})
);

//
 wire [160-1:0] arg_155_0_tdata;
 wire  arg_155_0_tvalid;
 wire  arg_155_0_tready;

//
 wire [160-1:0] arg_155_1_tdata;
 wire  arg_155_1_tvalid;
 wire  arg_155_1_tready;

//
 wire [160-1:0] arg_155_2_tdata;
 wire  arg_155_2_tvalid;
 wire  arg_155_2_tready;

//
 wire [160-1:0] arg_155_3_tdata;
 wire  arg_155_3_tvalid;
 wire  arg_155_3_tready;

axis_replication#(
.DATA_WIDTH(160),
.IF_STREAM(0),
.REAPLICA_COUNT(4)
)axis_replication_160(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_155_tdata}),
	.s_axis_in_tvalid({arg_155_tvalid}),
	.s_axis_in_tready({arg_155_tready}),
	//
	.m_axis_out_tdata({arg_155_0_tdata,arg_155_1_tdata,arg_155_2_tdata,arg_155_3_tdata}),
	.m_axis_out_tvalid({arg_155_0_tvalid,arg_155_1_tvalid,arg_155_2_tvalid,arg_155_3_tvalid}),
	.m_axis_out_tready({arg_155_0_tready,arg_155_1_tready,arg_155_2_tready,arg_155_3_tready})
);

//const_INT
 wire [16-1:0] const_INT_161_tdata=64;
 wire  const_INT_161_tvalid=1;
 wire  const_INT_161_tready;

//inited_BUF
 wire [512-1:0] inited_BUF_162_tdata=0;
 wire [64-1:0] inited_BUF_162_tkeep=0;
 wire  inited_BUF_162_tvalid=1;
 wire  inited_BUF_162_tready;
 wire  inited_BUF_162_tlast=1;

//Access Struct
 wire [48-1:0] struct_accessed_INT_164_tdata;
 wire  struct_accessed_INT_164_tvalid;
 wire  struct_accessed_INT_164_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(48)
)struct_access_163(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_154_0_tdata}),
	.s_struct_axis_tvalid({arg_154_0_tvalid}),
	.s_struct_axis_tready({arg_154_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_164_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_164_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_164_tready})
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_166_tdata;
 wire  struct_accessed_INT_166_tvalid;
 wire  struct_accessed_INT_166_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(48),
.ACCESS_SIZE(48)
)struct_access_165(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_154_1_tdata}),
	.s_struct_axis_tvalid({arg_154_1_tvalid}),
	.s_struct_axis_tready({arg_154_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_166_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_166_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_166_tready})
);

//struct_assign_167 output struct
 wire [112-1:0] structvar_168_tdata;
 wire  structvar_168_tvalid;
 wire  structvar_168_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(48)
)struct_assign_167(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({arg_154_2_tdata}),
	.s_struct_axis_tvalid({arg_154_2_tvalid}),
	.s_struct_axis_tready({arg_154_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_166_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_166_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_166_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_168_tdata}),
	.m_struct_axis_tvalid({structvar_168_tvalid}),
	.m_struct_axis_tready({structvar_168_tready})
);

//struct_assign_169 output struct
 wire [112-1:0] structvar_170_tdata;
 wire  structvar_170_tvalid;
 wire  structvar_170_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(48),
.ASSIGN_SIZE(48)
)struct_assign_169(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_168_tdata}),
	.s_struct_axis_tvalid({structvar_168_tvalid}),
	.s_struct_axis_tready({structvar_168_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_164_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_164_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_164_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_170_tdata}),
	.m_struct_axis_tvalid({structvar_170_tvalid}),
	.m_struct_axis_tready({structvar_170_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_172_tdata;
 wire  struct_accessed_INT_172_tvalid;
 wire  struct_accessed_INT_172_tready;

struct_access#(
.STRUCT_WIDTH(184),
.ACCESS_OFFSET(88),
.ACCESS_SIZE(32)
)struct_access_171(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_153_0_tdata}),
	.s_struct_axis_tvalid({arg_153_0_tvalid}),
	.s_struct_axis_tready({arg_153_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_172_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_172_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_172_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_174_tdata;
 wire  struct_accessed_INT_174_tvalid;
 wire  struct_accessed_INT_174_tready;

struct_access#(
.STRUCT_WIDTH(184),
.ACCESS_OFFSET(120),
.ACCESS_SIZE(32)
)struct_access_173(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_153_1_tdata}),
	.s_struct_axis_tvalid({arg_153_1_tvalid}),
	.s_struct_axis_tready({arg_153_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_174_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_174_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_174_tready})
);

//struct_assign_175 output struct
 wire [184-1:0] structvar_176_tdata;
 wire  structvar_176_tvalid;
 wire  structvar_176_tready;

struct_assign#(
.STRUCT_WIDTH(184),
.ASSIGN_OFFSET(88),
.ASSIGN_SIZE(32)
)struct_assign_175(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({arg_153_2_tdata}),
	.s_struct_axis_tvalid({arg_153_2_tvalid}),
	.s_struct_axis_tready({arg_153_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_174_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_174_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_174_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_176_tdata}),
	.m_struct_axis_tvalid({structvar_176_tvalid}),
	.m_struct_axis_tready({structvar_176_tready})
);

//struct_assign_177 output struct
 wire [184-1:0] structvar_178_tdata;
 wire  structvar_178_tvalid;
 wire  structvar_178_tready;

struct_assign#(
.STRUCT_WIDTH(184),
.ASSIGN_OFFSET(120),
.ASSIGN_SIZE(32)
)struct_assign_177(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_176_tdata}),
	.s_struct_axis_tvalid({structvar_176_tvalid}),
	.s_struct_axis_tready({structvar_176_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_172_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_172_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_172_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_178_tdata}),
	.m_struct_axis_tvalid({structvar_178_tvalid}),
	.m_struct_axis_tready({structvar_178_tready})
);

//struct_assign_179 output struct
 wire [184-1:0] structvar_180_tdata;
 wire  structvar_180_tvalid;
 wire  structvar_180_tready;

struct_assign#(
.STRUCT_WIDTH(184),
.ASSIGN_OFFSET(8),
.ASSIGN_SIZE(16)
)struct_assign_179(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_178_tdata}),
	.s_struct_axis_tvalid({structvar_178_tvalid}),
	.s_struct_axis_tready({structvar_178_tready}),
	//input val
	.s_assignv_axis_tdata({const_INT_161_tdata}),
	.s_assignv_axis_tvalid({const_INT_161_tvalid}),
	.s_assignv_axis_tready({const_INT_161_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_180_tdata}),
	.m_struct_axis_tvalid({structvar_180_tvalid}),
	.m_struct_axis_tready({structvar_180_tready})
);

//Access Struct
 wire [16-1:0] struct_accessed_INT_182_tdata;
 wire  struct_accessed_INT_182_tvalid;
 wire  struct_accessed_INT_182_tready;

struct_access#(
.STRUCT_WIDTH(160),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(16)
)struct_access_181(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_155_0_tdata}),
	.s_struct_axis_tvalid({arg_155_0_tvalid}),
	.s_struct_axis_tready({arg_155_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_182_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_182_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_182_tready})
);

//Access Struct
 wire [16-1:0] struct_accessed_INT_184_tdata;
 wire  struct_accessed_INT_184_tvalid;
 wire  struct_accessed_INT_184_tready;

struct_access#(
.STRUCT_WIDTH(160),
.ACCESS_OFFSET(16),
.ACCESS_SIZE(16)
)struct_access_183(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_155_1_tdata}),
	.s_struct_axis_tvalid({arg_155_1_tvalid}),
	.s_struct_axis_tready({arg_155_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_184_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_184_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_184_tready})
);

//struct_assign_185 output struct
 wire [160-1:0] structvar_186_tdata;
 wire  structvar_186_tvalid;
 wire  structvar_186_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(16)
)struct_assign_185(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({arg_155_2_tdata}),
	.s_struct_axis_tvalid({arg_155_2_tvalid}),
	.s_struct_axis_tready({arg_155_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_184_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_184_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_184_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_186_tdata}),
	.m_struct_axis_tvalid({structvar_186_tvalid}),
	.m_struct_axis_tready({structvar_186_tready})
);

//struct_assign_187 output struct
 wire [160-1:0] structvar_188_tdata;
 wire  structvar_188_tvalid;
 wire  structvar_188_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(16),
.ASSIGN_SIZE(16)
)struct_assign_187(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_186_tdata}),
	.s_struct_axis_tvalid({structvar_186_tvalid}),
	.s_struct_axis_tready({structvar_186_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_182_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_182_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_182_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_188_tdata}),
	.m_struct_axis_tvalid({structvar_188_tvalid}),
	.m_struct_axis_tready({structvar_188_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_190_tdata;
 wire  struct_accessed_INT_190_tvalid;
 wire  struct_accessed_INT_190_tready;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(32)
)struct_access_189(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_151_0_tdata}),
	.s_struct_axis_tvalid({arg_151_0_tvalid}),
	.s_struct_axis_tready({arg_151_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_190_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_190_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_190_tready})
);

//struct_assign_191 output struct
 wire [160-1:0] structvar_192_tdata;
 wire  structvar_192_tvalid;
 wire  structvar_192_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(32),
.ASSIGN_SIZE(32)
)struct_assign_191(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_188_tdata}),
	.s_struct_axis_tvalid({structvar_188_tvalid}),
	.s_struct_axis_tready({structvar_188_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_190_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_190_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_190_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_192_tdata}),
	.m_struct_axis_tvalid({structvar_192_tvalid}),
	.m_struct_axis_tready({structvar_192_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_194_tdata;
 wire  struct_accessed_INT_194_tvalid;
 wire  struct_accessed_INT_194_tready;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(32),
.ACCESS_SIZE(32)
)struct_access_193(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_151_1_tdata}),
	.s_struct_axis_tvalid({arg_151_1_tvalid}),
	.s_struct_axis_tready({arg_151_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_194_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_194_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_194_tready})
);

//struct_assign_195 output struct
 wire [160-1:0] structvar_196_tdata;
 wire  structvar_196_tvalid;
 wire  structvar_196_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(64),
.ASSIGN_SIZE(32)
)struct_assign_195(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_192_tdata}),
	.s_struct_axis_tvalid({structvar_192_tvalid}),
	.s_struct_axis_tready({structvar_192_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_194_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_194_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_194_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_196_tdata}),
	.m_struct_axis_tvalid({structvar_196_tvalid}),
	.m_struct_axis_tready({structvar_196_tready})
);

//emit_module_197 output buf
 wire [512-1:0] bufvar_198_tdata;
 wire [64-1:0] bufvar_198_tkeep;
 wire  bufvar_198_tvalid;
 wire  bufvar_198_tready;
 wire  bufvar_198_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(112)
)emit_module_197(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({inited_BUF_162_tdata}),
	.s_inbuf_axis_tkeep({inited_BUF_162_tkeep}),
	.s_inbuf_axis_tlast({inited_BUF_162_tlast}),
	.s_inbuf_axis_tvalid({inited_BUF_162_tvalid}),
	.s_inbuf_axis_tready({inited_BUF_162_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_170_tdata}),
	.s_struct_axis_tvalid({structvar_170_tvalid}),
	.s_struct_axis_tready({structvar_170_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_198_tdata}),
	.m_outbuf_axis_tkeep({bufvar_198_tkeep}),
	.m_outbuf_axis_tlast({bufvar_198_tlast}),
	.m_outbuf_axis_tvalid({bufvar_198_tvalid}),
	.m_outbuf_axis_tready({bufvar_198_tready})
);

//emit_module_199 output buf
 wire [512-1:0] bufvar_200_tdata;
 wire [64-1:0] bufvar_200_tkeep;
 wire  bufvar_200_tvalid;
 wire  bufvar_200_tready;
 wire  bufvar_200_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(184)
)emit_module_199(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_198_tdata}),
	.s_inbuf_axis_tkeep({bufvar_198_tkeep}),
	.s_inbuf_axis_tlast({bufvar_198_tlast}),
	.s_inbuf_axis_tvalid({bufvar_198_tvalid}),
	.s_inbuf_axis_tready({bufvar_198_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_180_tdata}),
	.s_struct_axis_tvalid({structvar_180_tvalid}),
	.s_struct_axis_tready({structvar_180_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_200_tdata}),
	.m_outbuf_axis_tkeep({bufvar_200_tkeep}),
	.m_outbuf_axis_tlast({bufvar_200_tlast}),
	.m_outbuf_axis_tvalid({bufvar_200_tvalid}),
	.m_outbuf_axis_tready({bufvar_200_tready})
);

//emit_module_201 output buf
 wire [512-1:0] bufvar_202_tdata;
 wire [64-1:0] bufvar_202_tkeep;
 wire  bufvar_202_tvalid;
 wire  bufvar_202_tready;
 wire  bufvar_202_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(160)
)emit_module_201(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_200_tdata}),
	.s_inbuf_axis_tkeep({bufvar_200_tkeep}),
	.s_inbuf_axis_tlast({bufvar_200_tlast}),
	.s_inbuf_axis_tvalid({bufvar_200_tvalid}),
	.s_inbuf_axis_tready({bufvar_200_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_196_tdata}),
	.s_struct_axis_tvalid({structvar_196_tvalid}),
	.s_struct_axis_tready({structvar_196_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_202_tdata}),
	.m_outbuf_axis_tkeep({bufvar_202_tkeep}),
	.m_outbuf_axis_tlast({bufvar_202_tlast}),
	.m_outbuf_axis_tvalid({bufvar_202_tvalid}),
	.m_outbuf_axis_tready({bufvar_202_tready})
);

//emit_module_203 output buf
 wire [512-1:0] bufvar_204_tdata;
 wire [64-1:0] bufvar_204_tkeep;
 wire  bufvar_204_tvalid;
 wire  bufvar_204_tready;
 wire  bufvar_204_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(1),
.INPUT_BUF_STRUCT_WIDTH(512)
)emit_module_203(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_202_tdata}),
	.s_inbuf_axis_tkeep({bufvar_202_tkeep}),
	.s_inbuf_axis_tlast({bufvar_202_tlast}),
	.s_inbuf_axis_tvalid({bufvar_202_tvalid}),
	.s_inbuf_axis_tready({bufvar_202_tready}),
	//input struct/buf
	.s_struct_axis_tdata({arg_152_0_tdata}),
	.s_struct_axis_tkeep({arg_152_0_tkeep}),
	.s_struct_axis_tlast({arg_152_0_tlast}),
	.s_struct_axis_tvalid({arg_152_0_tvalid}),
	.s_struct_axis_tready({arg_152_0_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_204_tdata}),
	.m_outbuf_axis_tkeep({bufvar_204_tkeep}),
	.m_outbuf_axis_tlast({bufvar_204_tlast}),
	.m_outbuf_axis_tvalid({bufvar_204_tvalid}),
	.m_outbuf_axis_tready({bufvar_204_tready})
);

 assign NET_SEND_0_tdata = bufvar_204_tdata;
 assign NET_SEND_0_tvalid = bufvar_204_tvalid;
 assign bufvar_204_tready = NET_SEND_0_tready;
 assign NET_SEND_0_tkeep = bufvar_204_tkeep;
 assign NET_SEND_0_tlast = bufvar_204_tlast;

 assign NET_SEND_1_tdata = arg_152_1_tdata;
 assign NET_SEND_1_tvalid = arg_152_1_tvalid;
 assign arg_152_1_tready = NET_SEND_1_tready;
 assign NET_SEND_1_tkeep = arg_152_1_tkeep;
 assign NET_SEND_1_tlast = arg_152_1_tlast;

 assign NET_SEND_2_tdata = arg_153_3_tdata;
 assign NET_SEND_2_tvalid = arg_153_3_tvalid;
 assign arg_153_3_tready = NET_SEND_2_tready;

 assign NET_SEND_3_tdata = arg_154_3_tdata;
 assign NET_SEND_3_tvalid = arg_154_3_tvalid;
 assign arg_154_3_tready = NET_SEND_3_tready;

 assign NET_SEND_4_tdata = arg_155_3_tdata;
 assign NET_SEND_4_tvalid = arg_155_3_tvalid;
 assign arg_155_3_tready = NET_SEND_4_tready;


endmodule
