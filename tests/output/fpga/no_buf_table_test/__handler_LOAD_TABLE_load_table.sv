module __handler_LOAD_TABLE_load_table#()
(
	 input  wire clk, 
	 input  wire rst
);
//Table lookup port wire def 
 wire [16-1:0] lookup_p_0_req_index;
 wire  lookup_p_0_req_valid;
 wire  lookup_p_0_req_ready;
 wire  lookup_p_0_value_valid;
 wire  lookup_p_0_value_ready;
 wire [32-1:0] lookup_p_0_value_data;

//Table lookup port wire def 
 wire [16-1:0] lookup_p_1_req_index;
 wire  lookup_p_1_req_valid;
 wire  lookup_p_1_req_ready;
 wire  lookup_p_1_value_valid;
 wire  lookup_p_1_value_ready;
 wire [32-1:0] lookup_p_1_value_data;

cam_arbiter#(
.TABLE_SIZE(16),
.KEY_SIZE(16),
.VALUE_SIZE(32),
.LOOKUP_PORTS(2),
.UPDATE_PORTS(1)
)table_0(
	 .clk(clk), 
	 .rst(rst) ,
	//lookup port 
	.s_lookup_req_index({lookup_p_0_req_index,lookup_p_1_req_index}),
	.s_lookup_req_valid({lookup_p_0_req_valid,lookup_p_1_req_valid}),
	.s_lookup_req_ready({lookup_p_0_req_ready,lookup_p_1_req_ready}),
	.s_lookup_value_valid({lookup_p_0_value_valid,lookup_p_1_value_valid}),
	.s_lookup_value_data({lookup_p_0_value_data,lookup_p_1_value_data}),
	.s_lookup_value_ready({lookup_p_0_value_ready,lookup_p_1_value_ready}),
	//update port 
	.s_update_req_index(),
	.s_update_req_index_valid(),
	.s_update_req_index_ready(),
	.s_update_req_data(),
	.s_update_req_data_valid(),
	.s_update_req_data_ready()
);

//const_INT
 wire [64-1:0] const_INT_1_tdata=10;
 wire  const_INT_1_tvalid=1;
 wire  const_INT_1_tready;

 assign lookup_p_1_req_index = const_INT_1_tdata;
 assign lookup_p_1_req_valid = const_INT_1_tvalid;
 assign const_INT_1_tready = lookup_p_1_req_ready;

//table lookup resultlookedup_INT_2
 wire [32-1:0] lookedup_INT_2_tdata;
 wire  lookedup_INT_2_tvalid;
 wire  lookedup_INT_2_tready=1;

 assign lookedup_INT_2_tdata = lookup_p_1_value_data;
 assign lookedup_INT_2_tvalid = lookup_p_1_value_valid;
 assign lookup_p_1_value_ready = lookedup_INT_2_tready;

//const_INT
 wire [64-1:0] const_INT_3_tdata=4;
 wire  const_INT_3_tvalid=1;
 wire  const_INT_3_tready;

 assign lookup_p_0_req_index = const_INT_3_tdata;
 assign lookup_p_0_req_valid = const_INT_3_tvalid;
 assign const_INT_3_tready = lookup_p_0_req_ready;

//table lookup resultlookedup_INT_4
 wire [32-1:0] lookedup_INT_4_tdata;
 wire  lookedup_INT_4_tvalid;
 wire  lookedup_INT_4_tready=1;

 assign lookedup_INT_4_tdata = lookup_p_0_value_data;
 assign lookedup_INT_4_tvalid = lookup_p_0_value_valid;
 assign lookup_p_0_value_ready = lookedup_INT_4_tready;


endmodule
