module __handler_ACK_GEN_ack_gen#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports STRUCT
	input wire [96-1:0] arg_126_tdata ,
	input wire  arg_126_tvalid ,
	output wire  arg_126_tready,
	//input ports BUF
	input wire [512-1:0] arg_127_tdata ,
	input wire [64-1:0] arg_127_tkeep ,
	input wire  arg_127_tlast ,
	input wire  arg_127_tvalid ,
	output wire  arg_127_tready,
	//input ports STRUCT
	input wire [184-1:0] arg_128_tdata ,
	input wire  arg_128_tvalid ,
	output wire  arg_128_tready,
	//input ports STRUCT
	input wire [112-1:0] arg_129_tdata ,
	input wire  arg_129_tvalid ,
	output wire  arg_129_tready,
	//input ports STRUCT
	input wire [160-1:0] arg_130_tdata ,
	input wire  arg_130_tvalid ,
	output wire  arg_130_tready,
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
 wire [96-1:0] arg_126_0_tdata;
 wire  arg_126_0_tvalid;
 wire  arg_126_0_tready;

//
 wire [96-1:0] arg_126_1_tdata;
 wire  arg_126_1_tvalid;
 wire  arg_126_1_tready;

axis_replication#(
.DATA_WIDTH(96),
.IF_STREAM(0),
.REAPLICA_COUNT(2)
)axis_replication_131(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_126_tdata}),
	.s_axis_in_tvalid({arg_126_tvalid}),
	.s_axis_in_tready({arg_126_tready}),
	//
	.m_axis_out_tdata({arg_126_0_tdata,arg_126_1_tdata}),
	.m_axis_out_tvalid({arg_126_0_tvalid,arg_126_1_tvalid}),
	.m_axis_out_tready({arg_126_0_tready,arg_126_1_tready})
);

//
 wire [512-1:0] arg_127_0_tdata;
 wire [64-1:0] arg_127_0_tkeep;
 wire  arg_127_0_tvalid;
 wire  arg_127_0_tready;
 wire  arg_127_0_tlast;

//
 wire [512-1:0] arg_127_1_tdata;
 wire [64-1:0] arg_127_1_tkeep;
 wire  arg_127_1_tvalid;
 wire  arg_127_1_tready;
 wire  arg_127_1_tlast;

axis_replication#(
.DATA_WIDTH(512),
.IF_STREAM(1),
.REAPLICA_COUNT(2)
)axis_replication_132(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_127_tdata}),
	.s_axis_in_tkeep({arg_127_tkeep}),
	.s_axis_in_tlast({arg_127_tlast}),
	.s_axis_in_tvalid({arg_127_tvalid}),
	.s_axis_in_tready({arg_127_tready}),
	//
	.m_axis_out_tdata({arg_127_0_tdata,arg_127_1_tdata}),
	.m_axis_out_tkeep({arg_127_0_tkeep,arg_127_1_tkeep}),
	.m_axis_out_tlast({arg_127_0_tlast,arg_127_1_tlast}),
	.m_axis_out_tvalid({arg_127_0_tvalid,arg_127_1_tvalid}),
	.m_axis_out_tready({arg_127_0_tready,arg_127_1_tready})
);

//
 wire [184-1:0] arg_128_0_tdata;
 wire  arg_128_0_tvalid;
 wire  arg_128_0_tready;

//
 wire [184-1:0] arg_128_1_tdata;
 wire  arg_128_1_tvalid;
 wire  arg_128_1_tready;

//
 wire [184-1:0] arg_128_2_tdata;
 wire  arg_128_2_tvalid;
 wire  arg_128_2_tready;

//
 wire [184-1:0] arg_128_3_tdata;
 wire  arg_128_3_tvalid;
 wire  arg_128_3_tready;

axis_replication#(
.DATA_WIDTH(184),
.IF_STREAM(0),
.REAPLICA_COUNT(4)
)axis_replication_133(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_128_tdata}),
	.s_axis_in_tvalid({arg_128_tvalid}),
	.s_axis_in_tready({arg_128_tready}),
	//
	.m_axis_out_tdata({arg_128_0_tdata,arg_128_1_tdata,arg_128_2_tdata,arg_128_3_tdata}),
	.m_axis_out_tvalid({arg_128_0_tvalid,arg_128_1_tvalid,arg_128_2_tvalid,arg_128_3_tvalid}),
	.m_axis_out_tready({arg_128_0_tready,arg_128_1_tready,arg_128_2_tready,arg_128_3_tready})
);

//
 wire [112-1:0] arg_129_0_tdata;
 wire  arg_129_0_tvalid;
 wire  arg_129_0_tready;

//
 wire [112-1:0] arg_129_1_tdata;
 wire  arg_129_1_tvalid;
 wire  arg_129_1_tready;

//
 wire [112-1:0] arg_129_2_tdata;
 wire  arg_129_2_tvalid;
 wire  arg_129_2_tready;

//
 wire [112-1:0] arg_129_3_tdata;
 wire  arg_129_3_tvalid;
 wire  arg_129_3_tready;

axis_replication#(
.DATA_WIDTH(112),
.IF_STREAM(0),
.REAPLICA_COUNT(4)
)axis_replication_134(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_129_tdata}),
	.s_axis_in_tvalid({arg_129_tvalid}),
	.s_axis_in_tready({arg_129_tready}),
	//
	.m_axis_out_tdata({arg_129_0_tdata,arg_129_1_tdata,arg_129_2_tdata,arg_129_3_tdata}),
	.m_axis_out_tvalid({arg_129_0_tvalid,arg_129_1_tvalid,arg_129_2_tvalid,arg_129_3_tvalid}),
	.m_axis_out_tready({arg_129_0_tready,arg_129_1_tready,arg_129_2_tready,arg_129_3_tready})
);

//
 wire [160-1:0] arg_130_0_tdata;
 wire  arg_130_0_tvalid;
 wire  arg_130_0_tready;

//
 wire [160-1:0] arg_130_1_tdata;
 wire  arg_130_1_tvalid;
 wire  arg_130_1_tready;

//
 wire [160-1:0] arg_130_2_tdata;
 wire  arg_130_2_tvalid;
 wire  arg_130_2_tready;

//
 wire [160-1:0] arg_130_3_tdata;
 wire  arg_130_3_tvalid;
 wire  arg_130_3_tready;

axis_replication#(
.DATA_WIDTH(160),
.IF_STREAM(0),
.REAPLICA_COUNT(4)
)axis_replication_135(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({arg_130_tdata}),
	.s_axis_in_tvalid({arg_130_tvalid}),
	.s_axis_in_tready({arg_130_tready}),
	//
	.m_axis_out_tdata({arg_130_0_tdata,arg_130_1_tdata,arg_130_2_tdata,arg_130_3_tdata}),
	.m_axis_out_tvalid({arg_130_0_tvalid,arg_130_1_tvalid,arg_130_2_tvalid,arg_130_3_tvalid}),
	.m_axis_out_tready({arg_130_0_tready,arg_130_1_tready,arg_130_2_tready,arg_130_3_tready})
);

//const_INT
 wire [16-1:0] const_INT_136_tdata=64;
 wire  const_INT_136_tvalid=1;
 wire  const_INT_136_tready;

//inited_BUF
 wire [512-1:0] inited_BUF_137_tdata=0;
 wire [64-1:0] inited_BUF_137_tkeep=0;
 wire  inited_BUF_137_tvalid=1;
 wire  inited_BUF_137_tready;
 wire  inited_BUF_137_tlast=1;

//Access Struct
 wire [48-1:0] struct_accessed_INT_139_tdata;
 wire  struct_accessed_INT_139_tvalid;
 wire  struct_accessed_INT_139_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(48)
)struct_access_138(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_129_0_tdata}),
	.s_struct_axis_tvalid({arg_129_0_tvalid}),
	.s_struct_axis_tready({arg_129_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_139_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_139_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_139_tready})
);

//Access Struct
 wire [48-1:0] struct_accessed_INT_141_tdata;
 wire  struct_accessed_INT_141_tvalid;
 wire  struct_accessed_INT_141_tready;

struct_access#(
.STRUCT_WIDTH(112),
.ACCESS_OFFSET(48),
.ACCESS_SIZE(48)
)struct_access_140(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_129_1_tdata}),
	.s_struct_axis_tvalid({arg_129_1_tvalid}),
	.s_struct_axis_tready({arg_129_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_141_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_141_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_141_tready})
);

//struct_assign_142 output struct
 wire [112-1:0] structvar_143_tdata;
 wire  structvar_143_tvalid;
 wire  structvar_143_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(48)
)struct_assign_142(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({arg_129_2_tdata}),
	.s_struct_axis_tvalid({arg_129_2_tvalid}),
	.s_struct_axis_tready({arg_129_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_141_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_141_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_141_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_143_tdata}),
	.m_struct_axis_tvalid({structvar_143_tvalid}),
	.m_struct_axis_tready({structvar_143_tready})
);

//struct_assign_144 output struct
 wire [112-1:0] structvar_145_tdata;
 wire  structvar_145_tvalid;
 wire  structvar_145_tready;

struct_assign#(
.STRUCT_WIDTH(112),
.ASSIGN_OFFSET(48),
.ASSIGN_SIZE(48)
)struct_assign_144(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_143_tdata}),
	.s_struct_axis_tvalid({structvar_143_tvalid}),
	.s_struct_axis_tready({structvar_143_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_139_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_139_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_139_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_145_tdata}),
	.m_struct_axis_tvalid({structvar_145_tvalid}),
	.m_struct_axis_tready({structvar_145_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_147_tdata;
 wire  struct_accessed_INT_147_tvalid;
 wire  struct_accessed_INT_147_tready;

struct_access#(
.STRUCT_WIDTH(184),
.ACCESS_OFFSET(88),
.ACCESS_SIZE(32)
)struct_access_146(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_128_0_tdata}),
	.s_struct_axis_tvalid({arg_128_0_tvalid}),
	.s_struct_axis_tready({arg_128_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_147_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_147_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_147_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_149_tdata;
 wire  struct_accessed_INT_149_tvalid;
 wire  struct_accessed_INT_149_tready;

struct_access#(
.STRUCT_WIDTH(184),
.ACCESS_OFFSET(120),
.ACCESS_SIZE(32)
)struct_access_148(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_128_1_tdata}),
	.s_struct_axis_tvalid({arg_128_1_tvalid}),
	.s_struct_axis_tready({arg_128_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_149_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_149_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_149_tready})
);

//struct_assign_150 output struct
 wire [184-1:0] structvar_151_tdata;
 wire  structvar_151_tvalid;
 wire  structvar_151_tready;

struct_assign#(
.STRUCT_WIDTH(184),
.ASSIGN_OFFSET(88),
.ASSIGN_SIZE(32)
)struct_assign_150(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({arg_128_2_tdata}),
	.s_struct_axis_tvalid({arg_128_2_tvalid}),
	.s_struct_axis_tready({arg_128_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_149_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_149_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_149_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_151_tdata}),
	.m_struct_axis_tvalid({structvar_151_tvalid}),
	.m_struct_axis_tready({structvar_151_tready})
);

//struct_assign_152 output struct
 wire [184-1:0] structvar_153_tdata;
 wire  structvar_153_tvalid;
 wire  structvar_153_tready;

struct_assign#(
.STRUCT_WIDTH(184),
.ASSIGN_OFFSET(120),
.ASSIGN_SIZE(32)
)struct_assign_152(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_151_tdata}),
	.s_struct_axis_tvalid({structvar_151_tvalid}),
	.s_struct_axis_tready({structvar_151_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_147_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_147_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_147_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_153_tdata}),
	.m_struct_axis_tvalid({structvar_153_tvalid}),
	.m_struct_axis_tready({structvar_153_tready})
);

//struct_assign_154 output struct
 wire [184-1:0] structvar_155_tdata;
 wire  structvar_155_tvalid;
 wire  structvar_155_tready;

struct_assign#(
.STRUCT_WIDTH(184),
.ASSIGN_OFFSET(8),
.ASSIGN_SIZE(16)
)struct_assign_154(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_153_tdata}),
	.s_struct_axis_tvalid({structvar_153_tvalid}),
	.s_struct_axis_tready({structvar_153_tready}),
	//input val
	.s_assignv_axis_tdata({const_INT_136_tdata}),
	.s_assignv_axis_tvalid({const_INT_136_tvalid}),
	.s_assignv_axis_tready({const_INT_136_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_155_tdata}),
	.m_struct_axis_tvalid({structvar_155_tvalid}),
	.m_struct_axis_tready({structvar_155_tready})
);

//Access Struct
 wire [16-1:0] struct_accessed_INT_157_tdata;
 wire  struct_accessed_INT_157_tvalid;
 wire  struct_accessed_INT_157_tready;

struct_access#(
.STRUCT_WIDTH(160),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(16)
)struct_access_156(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_130_0_tdata}),
	.s_struct_axis_tvalid({arg_130_0_tvalid}),
	.s_struct_axis_tready({arg_130_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_157_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_157_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_157_tready})
);

//Access Struct
 wire [16-1:0] struct_accessed_INT_159_tdata;
 wire  struct_accessed_INT_159_tvalid;
 wire  struct_accessed_INT_159_tready;

struct_access#(
.STRUCT_WIDTH(160),
.ACCESS_OFFSET(16),
.ACCESS_SIZE(16)
)struct_access_158(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_130_1_tdata}),
	.s_struct_axis_tvalid({arg_130_1_tvalid}),
	.s_struct_axis_tready({arg_130_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_159_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_159_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_159_tready})
);

//struct_assign_160 output struct
 wire [160-1:0] structvar_161_tdata;
 wire  structvar_161_tvalid;
 wire  structvar_161_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(0),
.ASSIGN_SIZE(16)
)struct_assign_160(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({arg_130_2_tdata}),
	.s_struct_axis_tvalid({arg_130_2_tvalid}),
	.s_struct_axis_tready({arg_130_2_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_159_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_159_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_159_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_161_tdata}),
	.m_struct_axis_tvalid({structvar_161_tvalid}),
	.m_struct_axis_tready({structvar_161_tready})
);

//struct_assign_162 output struct
 wire [160-1:0] structvar_163_tdata;
 wire  structvar_163_tvalid;
 wire  structvar_163_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(16),
.ASSIGN_SIZE(16)
)struct_assign_162(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_161_tdata}),
	.s_struct_axis_tvalid({structvar_161_tvalid}),
	.s_struct_axis_tready({structvar_161_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_157_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_157_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_157_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_163_tdata}),
	.m_struct_axis_tvalid({structvar_163_tvalid}),
	.m_struct_axis_tready({structvar_163_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_165_tdata;
 wire  struct_accessed_INT_165_tvalid;
 wire  struct_accessed_INT_165_tready;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(32)
)struct_access_164(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_126_0_tdata}),
	.s_struct_axis_tvalid({arg_126_0_tvalid}),
	.s_struct_axis_tready({arg_126_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_165_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_165_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_165_tready})
);

//struct_assign_166 output struct
 wire [160-1:0] structvar_167_tdata;
 wire  structvar_167_tvalid;
 wire  structvar_167_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(32),
.ASSIGN_SIZE(32)
)struct_assign_166(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_163_tdata}),
	.s_struct_axis_tvalid({structvar_163_tvalid}),
	.s_struct_axis_tready({structvar_163_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_165_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_165_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_165_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_167_tdata}),
	.m_struct_axis_tvalid({structvar_167_tvalid}),
	.m_struct_axis_tready({structvar_167_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_169_tdata;
 wire  struct_accessed_INT_169_tvalid;
 wire  struct_accessed_INT_169_tready;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(32),
.ACCESS_SIZE(32)
)struct_access_168(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_126_1_tdata}),
	.s_struct_axis_tvalid({arg_126_1_tvalid}),
	.s_struct_axis_tready({arg_126_1_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_169_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_169_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_169_tready})
);

//struct_assign_170 output struct
 wire [160-1:0] structvar_171_tdata;
 wire  structvar_171_tvalid;
 wire  structvar_171_tready;

struct_assign#(
.STRUCT_WIDTH(160),
.ASSIGN_OFFSET(64),
.ASSIGN_SIZE(32)
)struct_assign_170(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_167_tdata}),
	.s_struct_axis_tvalid({structvar_167_tvalid}),
	.s_struct_axis_tready({structvar_167_tready}),
	//input val
	.s_assignv_axis_tdata({struct_accessed_INT_169_tdata}),
	.s_assignv_axis_tvalid({struct_accessed_INT_169_tvalid}),
	.s_assignv_axis_tready({struct_accessed_INT_169_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_171_tdata}),
	.m_struct_axis_tvalid({structvar_171_tvalid}),
	.m_struct_axis_tready({structvar_171_tready})
);

//emit_module_172 output buf
 wire [512-1:0] bufvar_173_tdata;
 wire [64-1:0] bufvar_173_tkeep;
 wire  bufvar_173_tvalid;
 wire  bufvar_173_tready;
 wire  bufvar_173_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(112)
)emit_module_172(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({inited_BUF_137_tdata}),
	.s_inbuf_axis_tkeep({inited_BUF_137_tkeep}),
	.s_inbuf_axis_tlast({inited_BUF_137_tlast}),
	.s_inbuf_axis_tvalid({inited_BUF_137_tvalid}),
	.s_inbuf_axis_tready({inited_BUF_137_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_145_tdata}),
	.s_struct_axis_tvalid({structvar_145_tvalid}),
	.s_struct_axis_tready({structvar_145_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_173_tdata}),
	.m_outbuf_axis_tkeep({bufvar_173_tkeep}),
	.m_outbuf_axis_tlast({bufvar_173_tlast}),
	.m_outbuf_axis_tvalid({bufvar_173_tvalid}),
	.m_outbuf_axis_tready({bufvar_173_tready})
);

//emit_module_174 output buf
 wire [512-1:0] bufvar_175_tdata;
 wire [64-1:0] bufvar_175_tkeep;
 wire  bufvar_175_tvalid;
 wire  bufvar_175_tready;
 wire  bufvar_175_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(184)
)emit_module_174(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_173_tdata}),
	.s_inbuf_axis_tkeep({bufvar_173_tkeep}),
	.s_inbuf_axis_tlast({bufvar_173_tlast}),
	.s_inbuf_axis_tvalid({bufvar_173_tvalid}),
	.s_inbuf_axis_tready({bufvar_173_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_155_tdata}),
	.s_struct_axis_tvalid({structvar_155_tvalid}),
	.s_struct_axis_tready({structvar_155_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_175_tdata}),
	.m_outbuf_axis_tkeep({bufvar_175_tkeep}),
	.m_outbuf_axis_tlast({bufvar_175_tlast}),
	.m_outbuf_axis_tvalid({bufvar_175_tvalid}),
	.m_outbuf_axis_tready({bufvar_175_tready})
);

//emit_module_176 output buf
 wire [512-1:0] bufvar_177_tdata;
 wire [64-1:0] bufvar_177_tkeep;
 wire  bufvar_177_tvalid;
 wire  bufvar_177_tready;
 wire  bufvar_177_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(0),
.INPUT_BUF_STRUCT_WIDTH(160)
)emit_module_176(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_175_tdata}),
	.s_inbuf_axis_tkeep({bufvar_175_tkeep}),
	.s_inbuf_axis_tlast({bufvar_175_tlast}),
	.s_inbuf_axis_tvalid({bufvar_175_tvalid}),
	.s_inbuf_axis_tready({bufvar_175_tready}),
	//input struct/buf
	.s_struct_axis_tdata({structvar_171_tdata}),
	.s_struct_axis_tvalid({structvar_171_tvalid}),
	.s_struct_axis_tready({structvar_171_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_177_tdata}),
	.m_outbuf_axis_tkeep({bufvar_177_tkeep}),
	.m_outbuf_axis_tlast({bufvar_177_tlast}),
	.m_outbuf_axis_tvalid({bufvar_177_tvalid}),
	.m_outbuf_axis_tready({bufvar_177_tready})
);

//emit_module_178 output buf
 wire [512-1:0] bufvar_179_tdata;
 wire [64-1:0] bufvar_179_tkeep;
 wire  bufvar_179_tvalid;
 wire  bufvar_179_tready;
 wire  bufvar_179_tlast;

emit#(
.BUF_DATA_WIDTH(512),
.BUF_KEEP_WIDTH(64),
.IF_INPUT_BUF(1),
.INPUT_BUF_STRUCT_WIDTH(512)
)emit_module_178(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.s_inbuf_axis_tdata({bufvar_177_tdata}),
	.s_inbuf_axis_tkeep({bufvar_177_tkeep}),
	.s_inbuf_axis_tlast({bufvar_177_tlast}),
	.s_inbuf_axis_tvalid({bufvar_177_tvalid}),
	.s_inbuf_axis_tready({bufvar_177_tready}),
	//input struct/buf
	.s_struct_axis_tdata({arg_127_0_tdata}),
	.s_struct_axis_tkeep({arg_127_0_tkeep}),
	.s_struct_axis_tlast({arg_127_0_tlast}),
	.s_struct_axis_tvalid({arg_127_0_tvalid}),
	.s_struct_axis_tready({arg_127_0_tready}),
	//output buf
	.m_outbuf_axis_tdata({bufvar_179_tdata}),
	.m_outbuf_axis_tkeep({bufvar_179_tkeep}),
	.m_outbuf_axis_tlast({bufvar_179_tlast}),
	.m_outbuf_axis_tvalid({bufvar_179_tvalid}),
	.m_outbuf_axis_tready({bufvar_179_tready})
);

 assign NET_SEND_0_tdata = bufvar_179_tdata;
 assign NET_SEND_0_tvalid = bufvar_179_tvalid;
 assign bufvar_179_tready = NET_SEND_0_tready;
 assign NET_SEND_0_tkeep = bufvar_179_tkeep;
 assign NET_SEND_0_tlast = bufvar_179_tlast;

 assign NET_SEND_1_tdata = arg_127_1_tdata;
 assign NET_SEND_1_tvalid = arg_127_1_tvalid;
 assign arg_127_1_tready = NET_SEND_1_tready;
 assign NET_SEND_1_tkeep = arg_127_1_tkeep;
 assign NET_SEND_1_tlast = arg_127_1_tlast;

 assign NET_SEND_2_tdata = arg_128_3_tdata;
 assign NET_SEND_2_tvalid = arg_128_3_tvalid;
 assign arg_128_3_tready = NET_SEND_2_tready;

 assign NET_SEND_3_tdata = arg_129_3_tdata;
 assign NET_SEND_3_tvalid = arg_129_3_tvalid;
 assign arg_129_3_tready = NET_SEND_3_tready;

 assign NET_SEND_4_tdata = arg_130_3_tdata;
 assign NET_SEND_4_tvalid = arg_130_3_tvalid;
 assign arg_130_3_tready = NET_SEND_4_tready;


endmodule
