module __handler_OoO_DETECT_ooo_detect#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports STRUCT
	input wire [96-1:0] arg_19_tdata ,
	input wire  arg_19_tvalid ,
	output wire  arg_19_tready
);
//Table lookup port wire def 
 wire [16-1:0] lookup_p_0_req_index;
 wire  lookup_p_0_req_valid;
 wire  lookup_p_0_req_ready;
 wire  lookup_p_0_value_valid;
 wire  lookup_p_0_value_ready;
 wire [256-1:0] lookup_p_0_value_data;

//Table update port wire def 
 wire [16-1:0] update_p_0_req_index;
 wire  update_p_0_req_index_valid;
 wire  update_p_0_req_index_ready;
 wire [256-1:0] update_p_0_req_data;
 wire  update_p_0_req_data_valid;
 wire  update_p_0_req_data_ready;

cam_arbiter#(
.TABLE_SIZE(16),
.KEY_SIZE(16),
.VALUE_SIZE(256),
.LOOKUP_PORTS(1),
.UPDATE_PORTS(1)
)table_20(
	 .clk(clk), 
	 .rst(rst) ,
	//lookup port 
	.s_lookup_req_index({lookup_p_0_req_index}),
	.s_lookup_req_valid({lookup_p_0_req_valid}),
	.s_lookup_req_ready({lookup_p_0_req_ready}),
	.s_lookup_value_valid({lookup_p_0_value_valid}),
	.s_lookup_value_data({lookup_p_0_value_data}),
	.s_lookup_value_ready({lookup_p_0_value_ready}),
	//update port 
	.s_update_req_index({update_p_0_req_index}),
	.s_update_req_index_valid({update_p_0_req_index_valid}),
	.s_update_req_index_ready({update_p_0_req_index_ready}),
	.s_update_req_data({update_p_0_req_data}),
	.s_update_req_data_valid({update_p_0_req_data_valid}),
	.s_update_req_data_ready({update_p_0_req_data_ready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_22_tdata;
 wire  struct_accessed_INT_22_tvalid;
 wire  struct_accessed_INT_22_tready;

//Struct Assign new Struct
 wire [96-1:0] structvar_23_tdata;
 wire  structvar_23_tvalid;
 wire  structvar_23_tready;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(32)
)struct_access_21(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({arg_19_tdata}),
	.s_struct_axis_tvalid({arg_19_tvalid}),
	.s_struct_axis_tready({arg_19_tready}),
	//struct output
	.m_struct_axis_tdata({structvar_23_tdata}),
	.m_struct_axis_tvalid({structvar_23_tvalid}),
	.m_struct_axis_tready({structvar_23_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_22_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_22_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_22_tready})
);

 assign lookup_p_0_req_index = struct_accessed_INT_22_tdata;
 assign lookup_p_0_req_valid = struct_accessed_INT_22_tvalid;
 assign struct_accessed_INT_22_tready = lookup_p_0_req_ready;

//table lookup resultlookedup_STRUCT_24
 wire [256-1:0] lookedup_STRUCT_24_tdata;
 wire  lookedup_STRUCT_24_tvalid;
 wire  lookedup_STRUCT_24_tready;

 assign lookedup_STRUCT_24_tdata = lookup_p_0_value_data;
 assign lookedup_STRUCT_24_tvalid = lookup_p_0_value_valid;
 assign lookup_p_0_value_ready = lookedup_STRUCT_24_tready;

//Access Struct
 wire [32-1:0] struct_accessed_INT_26_tdata;
 wire  struct_accessed_INT_26_tvalid;
 wire  struct_accessed_INT_26_tready;

//Struct Assign new Struct
 wire [256-1:0] structvar_27_tdata;
 wire  structvar_27_tvalid;
 wire  structvar_27_tready;

struct_access#(
.STRUCT_WIDTH(256),
.ACCESS_OFFSET(128),
.ACCESS_SIZE(32)
)struct_access_25(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({lookedup_STRUCT_24_tdata}),
	.s_struct_axis_tvalid({lookedup_STRUCT_24_tvalid}),
	.s_struct_axis_tready({lookedup_STRUCT_24_tready}),
	//struct output
	.m_struct_axis_tdata({structvar_27_tdata}),
	.m_struct_axis_tvalid({structvar_27_tvalid}),
	.m_struct_axis_tready({structvar_27_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_26_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_26_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_26_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_29_tdata;
 wire  struct_accessed_INT_29_tvalid;
 wire  struct_accessed_INT_29_tready;

//Struct Assign new Struct
 wire [96-1:0] structvar_30_tdata;
 wire  structvar_30_tvalid;
 wire  structvar_30_tready;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(32),
.ACCESS_SIZE(32)
)struct_access_28(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_23_tdata}),
	.s_struct_axis_tvalid({structvar_23_tvalid}),
	.s_struct_axis_tready({structvar_23_tready}),
	//struct output
	.m_struct_axis_tdata({structvar_30_tdata}),
	.m_struct_axis_tvalid({structvar_30_tvalid}),
	.m_struct_axis_tready({structvar_30_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_29_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_29_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_29_tready})
);

//Arithmetic OP Out
 wire [32-1:0] ADD_31_out_INT_32_tdata;
 wire  ADD_31_out_INT_32_tvalid;
 wire  ADD_31_out_INT_32_tready;

ALU#(
.LVAL_SIZE(32),
.RVAL_SIZE(32),
.RESULT_SIZE(32),
.OPID(1)
)ADD_31(
	 .clk(clk), 
	 .rst(rst) ,
	//lval input
	.s_lval_axis_tdata({struct_accessed_INT_26_tdata}),
	.s_lval_axis_tvalid({struct_accessed_INT_26_tvalid}),
	.s_lval_axis_tready({struct_accessed_INT_26_tready}),
	//rval input
	.s_rval_axis_tdata({struct_accessed_INT_29_tdata}),
	.s_rval_axis_tvalid({struct_accessed_INT_29_tvalid}),
	.s_rval_axis_tready({struct_accessed_INT_29_tready}),
	//output val
	.m_val_axis_tdata({ADD_31_out_INT_32_tdata}),
	.m_val_axis_tvalid({ADD_31_out_INT_32_tvalid}),
	.m_val_axis_tready({ADD_31_out_INT_32_tready})
);

//struct_assign_33 output struct
 wire [256-1:0] structvar_34_tdata;
 wire  structvar_34_tvalid;
 wire  structvar_34_tready;

struct_assign#(
.STRUCT_WIDTH(256),
.ASSIGN_OFFSET(128),
.ASSIGN_SIZE(32)
)struct_assign_33(
	 .clk(clk), 
	 .rst(rst) ,
	//input struct
	.s_struct_axis_tdata({structvar_27_tdata}),
	.s_struct_axis_tvalid({structvar_27_tvalid}),
	.s_struct_axis_tready({structvar_27_tready}),
	//input val
	.s_assignv_axis_tdata({ADD_31_out_INT_32_tdata}),
	.s_assignv_axis_tvalid({ADD_31_out_INT_32_tvalid}),
	.s_assignv_axis_tready({ADD_31_out_INT_32_tready}),
	//output struct
	.m_struct_axis_tdata({structvar_34_tdata}),
	.m_struct_axis_tvalid({structvar_34_tvalid}),
	.m_struct_axis_tready({structvar_34_tready})
);

//Access Struct
 wire [32-1:0] struct_accessed_INT_36_tdata;
 wire  struct_accessed_INT_36_tvalid;
 wire  struct_accessed_INT_36_tready;

//Struct Assign new Struct
 wire [96-1:0] structvar_37_tdata;
 wire  structvar_37_tvalid;
 wire  structvar_37_tready = 1;

struct_access#(
.STRUCT_WIDTH(96),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(32)
)struct_access_35(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({structvar_30_tdata}),
	.s_struct_axis_tvalid({structvar_30_tvalid}),
	.s_struct_axis_tready({structvar_30_tready}),
	//struct output
	.m_struct_axis_tdata({structvar_37_tdata}),
	.m_struct_axis_tvalid({structvar_37_tvalid}),
	.m_struct_axis_tready({structvar_37_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_36_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_36_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_36_tready})
);

 assign update_p_0_req_index = struct_accessed_INT_36_tdata;
 assign update_p_0_req_index_valid = struct_accessed_INT_36_tvalid;
 assign struct_accessed_INT_36_tready = update_p_0_req_index_ready;

 assign update_p_0_req_data = structvar_34_tdata;
 assign update_p_0_req_data_valid = structvar_34_tvalid;
 assign structvar_34_tready = update_p_0_req_data_ready;


endmodule
