module __handler_NET_RECV_process_packet#()
(
	 input  wire clk, 
	 input  wire rst,
	//input ports STRUCT
	input wire [1-1:0] NET_RECV_0_tdata ,
	input wire  NET_RECV_0_tvalid ,
	output wire  NET_RECV_0_tready,
	//output ports INT
	output wire [32-1:0] outport_0_0_tdata ,
	output wire  outport_0_0_tvalid ,
	input wire  outport_0_0_tready
);
//BB1pred_demux_in0
 wire [1-1:0] BB1pred_demux_in0_tdata;
 wire  BB1pred_demux_in0_tvalid;
 wire  BB1pred_demux_in0_tready;

//BB1pred_demux_out0_1
 wire [1-1:0] BB1pred_demux_out0_1_tdata;
 wire  BB1pred_demux_out0_1_tvalid;
 wire  BB1pred_demux_out0_1_tready;

pred_demux#(
.VAL_WIDTH(1),
.PORT_COUNT(1),
.IF_STREAM(0),
.IF_LOCAL_PRED_OUT(0)
)BB1demux_arg0(
	 .clk(clk), 
	 .rst(rst) ,
	//demux in port 
	.s_demux_in_tdata({BB1pred_demux_in0_tdata}),
	.s_demux_in_tvalid({BB1pred_demux_in0_tvalid}),
	.s_demux_in_tready({BB1pred_demux_in0_tready}),
	//demux out port 
	.m_demux_out_tdata({BB1pred_demux_out0_1_tdata}),
	.m_demux_out_tvalid({BB1pred_demux_out0_1_tvalid}),
	.m_demux_out_tready({BB1pred_demux_out0_1_tready})
);

//BB2pred_demux_in0
 wire [1-1:0] BB2pred_demux_in0_tdata;
 wire  BB2pred_demux_in0_tvalid;
 wire  BB2pred_demux_in0_tready;

//BB2pred_demux_in1
 wire [1-1:0] BB2pred_demux_in1_tdata;
 wire  BB2pred_demux_in1_tvalid;
 wire  BB2pred_demux_in1_tready;

//BB2pred_demux_out0_2
 wire [1-1:0] BB2pred_demux_out0_2_tdata;
 wire  BB2pred_demux_out0_2_tvalid;
 wire  BB2pred_demux_out0_2_tready;

//BB2pred_demux_out0_2
 wire [2-1:0] BB2pred_demux_out0_2_local_pred_0_tdata;
 wire  BB2pred_demux_out0_2_local_pred_0_tvalid;
 wire  BB2pred_demux_out0_2_local_pred_0_tready;

pred_demux#(
.VAL_WIDTH(1),
.PORT_COUNT(2),
.IF_STREAM(0),
.IF_LOCAL_PRED_OUT(1)
)BB2demux_arg0(
	 .clk(clk), 
	 .rst(rst) ,
	//demux in port 
	.s_demux_in_tdata({BB2pred_demux_in0_tdata,BB2pred_demux_in1_tdata}),
	.s_demux_in_tvalid({BB2pred_demux_in0_tvalid,BB2pred_demux_in1_tvalid}),
	.s_demux_in_tready({BB2pred_demux_in0_tready,BB2pred_demux_in1_tready}),
	//local pred out
	.m_pred_out_tdata({BB2pred_demux_out0_2_local_pred_0_tdata}),
	.m_pred_out_tvalid({BB2pred_demux_out0_2_local_pred_0_tvalid}),
	.m_pred_out_tready({BB2pred_demux_out0_2_local_pred_0_tready}),
	//demux out port 
	.m_demux_out_tdata({BB2pred_demux_out0_2_tdata}),
	.m_demux_out_tvalid({BB2pred_demux_out0_2_tvalid}),
	.m_demux_out_tready({BB2pred_demux_out0_2_tready})
);

//BB2arg_demux_in0
 wire [32-1:0] BB2arg_demux_in0_tdata;
 wire  BB2arg_demux_in0_tvalid;
 wire  BB2arg_demux_in0_tready;

//BB2arg_demux_in1
 wire [32-1:0] BB2arg_demux_in1_tdata;
 wire  BB2arg_demux_in1_tvalid;
 wire  BB2arg_demux_in1_tready;

//BB2arg_demux_out1_3
 wire [32-1:0] BB2arg_demux_out1_3_tdata;
 wire  BB2arg_demux_out1_3_tvalid;
 wire  BB2arg_demux_out1_3_tready;

arg_demux#(
.VAL_WIDTH(32),
.PORT_COUNT(2),
.IF_STREAM(0)
)BB2demux_arg1(
	 .clk(clk), 
	 .rst(rst) ,
	//demux in port 
	.s_demux_in_tdata({BB2arg_demux_in0_tdata,BB2arg_demux_in1_tdata}),
	.s_demux_in_tvalid({BB2arg_demux_in0_tvalid,BB2arg_demux_in1_tvalid}),
	.s_demux_in_tready({BB2arg_demux_in0_tready,BB2arg_demux_in1_tready}),
	//pred in port 
	.s_pred_in_tdata({BB2pred_demux_out0_2_local_pred_0_tdata}),
	.s_pred_in_tvalid({BB2pred_demux_out0_2_local_pred_0_tvalid}),
	.s_pred_in_tready({BB2pred_demux_out0_2_local_pred_0_tready}),
	//demux out port 
	.m_demux_out_tdata({BB2arg_demux_out1_3_tdata}),
	.m_demux_out_tvalid({BB2arg_demux_out1_3_tvalid}),
	.m_demux_out_tready({BB2arg_demux_out1_3_tready})
);

//const_INT
 wire [1-1:0] const_INT_4_tdata=-1;
 wire  const_INT_4_tvalid=1;
 wire  const_INT_4_tready;

//const_INT
 wire [32-1:0] const_INT_5_tdata=0;
 wire  const_INT_5_tvalid=1;
 wire  const_INT_5_tready;

//const_INT
 wire [32-1:0] const_INT_5_0_tdata;
 wire  const_INT_5_0_tvalid;
 wire  const_INT_5_0_tready;

//const_INT
 wire [32-1:0] const_INT_5_1_tdata;
 wire  const_INT_5_1_tvalid;
 wire  const_INT_5_1_tready;

axis_replication#(
.DATA_WIDTH(32),
.IF_STREAM(0),
.REAPLICA_COUNT(2)
)axis_replication_6(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({const_INT_5_tdata}),
	.s_axis_in_tvalid({const_INT_5_tvalid}),
	.s_axis_in_tready({const_INT_5_tready}),
	//
	.m_axis_out_tdata({const_INT_5_0_tdata,const_INT_5_1_tdata}),
	.m_axis_out_tvalid({const_INT_5_0_tvalid,const_INT_5_1_tvalid}),
	.m_axis_out_tready({const_INT_5_0_tready,const_INT_5_1_tready})
);

//const_INT
 wire [64-1:0] const_INT_7_tdata=1;
 wire  const_INT_7_tvalid=1;
 wire  const_INT_7_tready;

//Access Struct
 wire [1-1:0] struct_accessed_INT_9_tdata;
 wire  struct_accessed_INT_9_tvalid;
 wire  struct_accessed_INT_9_tready;

//Access Struct
 wire [1-1:0] struct_accessed_INT_9_0_tdata;
 wire  struct_accessed_INT_9_0_tvalid;
 wire  struct_accessed_INT_9_0_tready;

//Access Struct
 wire [1-1:0] struct_accessed_INT_9_1_tdata;
 wire  struct_accessed_INT_9_1_tvalid;
 wire  struct_accessed_INT_9_1_tready;

//Access Struct
 wire [1-1:0] struct_accessed_INT_9_2_tdata;
 wire  struct_accessed_INT_9_2_tvalid;
 wire  struct_accessed_INT_9_2_tready;

axis_replication#(
.DATA_WIDTH(1),
.IF_STREAM(0),
.REAPLICA_COUNT(3)
)axis_replication_10(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata({struct_accessed_INT_9_tdata}),
	.s_axis_in_tvalid({struct_accessed_INT_9_tvalid}),
	.s_axis_in_tready({struct_accessed_INT_9_tready}),
	//
	.m_axis_out_tdata({struct_accessed_INT_9_0_tdata,struct_accessed_INT_9_1_tdata,struct_accessed_INT_9_2_tdata}),
	.m_axis_out_tvalid({struct_accessed_INT_9_0_tvalid,struct_accessed_INT_9_1_tvalid,struct_accessed_INT_9_2_tvalid}),
	.m_axis_out_tready({struct_accessed_INT_9_0_tready,struct_accessed_INT_9_1_tready,struct_accessed_INT_9_2_tready})
);

struct_access#(
.STRUCT_WIDTH(1),
.ACCESS_OFFSET(0),
.ACCESS_SIZE(1)
)struct_access_8(
	 .clk(clk), 
	 .rst(rst) ,
	//struct input
	.s_struct_axis_tdata({NET_RECV_0_tdata}),
	.s_struct_axis_tvalid({NET_RECV_0_tvalid}),
	.s_struct_axis_tready({NET_RECV_0_tready}),
	//output val
	.m_val_axis_tdata({struct_accessed_INT_9_tdata}),
	.m_val_axis_tvalid({struct_accessed_INT_9_tvalid}),
	.m_val_axis_tready({struct_accessed_INT_9_tready})
);

//Arithmetic OP Out
 wire [1-1:0] SUB_11_out_INT_12_tdata;
 wire  SUB_11_out_INT_12_tvalid;
 wire  SUB_11_out_INT_12_tready;

ALU#(
.LVAL_SIZE(1),
.RVAL_SIZE(1),
.RESULT_SIZE(1),
.OPID(0)
)SUB_11(
	 .clk(clk), 
	 .rst(rst) ,
	//lval input
	.s_lval_axis_tdata({const_INT_4_tdata}),
	.s_lval_axis_tvalid({const_INT_4_tvalid}),
	.s_lval_axis_tready({const_INT_4_tready}),
	//rval input
	.s_rval_axis_tdata({struct_accessed_INT_9_0_tdata}),
	.s_rval_axis_tvalid({struct_accessed_INT_9_0_tvalid}),
	.s_rval_axis_tready({struct_accessed_INT_9_0_tready}),
	//output val
	.m_val_axis_tdata({SUB_11_out_INT_12_tdata}),
	.m_val_axis_tvalid({SUB_11_out_INT_12_tvalid}),
	.m_val_axis_tready({SUB_11_out_INT_12_tready})
);

 assign struct_accessed_INT_9_1_tready = 1;
 assign BB1pred_demux_in0_tdata = struct_accessed_INT_9_2_tdata;
 assign BB1pred_demux_in0_tvalid = struct_accessed_INT_9_2_tvalid;
 assign struct_accessed_INT_9_2_tready = BB1pred_demux_in0_tready;

 assign BB2pred_demux_in0_tdata = SUB_11_out_INT_12_tdata;
 assign BB2pred_demux_in0_tvalid = SUB_11_out_INT_12_tvalid;
 assign SUB_11_out_INT_12_tready = BB2pred_demux_in0_tready;

 assign BB2arg_demux_in0_tdata = const_INT_5_0_tdata;
 assign BB2arg_demux_in0_tvalid = const_INT_5_0_tvalid;
 assign const_INT_5_0_tready = BB2arg_demux_in0_tready;

//Arithmetic OP Out
 wire [32-1:0] ADD_13_out_INT_14_tdata;
 wire  ADD_13_out_INT_14_tvalid;
 wire  ADD_13_out_INT_14_tready;

ALU#(
.LVAL_SIZE(32),
.RVAL_SIZE(64),
.RESULT_SIZE(32),
.OPID(1)
)ADD_13(
	 .clk(clk), 
	 .rst(rst) ,
	//lval input
	.s_lval_axis_tdata({const_INT_5_1_tdata}),
	.s_lval_axis_tvalid({const_INT_5_1_tvalid}),
	.s_lval_axis_tready({const_INT_5_1_tready}),
	//rval input
	.s_rval_axis_tdata({const_INT_7_tdata}),
	.s_rval_axis_tvalid({const_INT_7_tvalid}),
	.s_rval_axis_tready({const_INT_7_tready}),
	//output val
	.m_val_axis_tdata({ADD_13_out_INT_14_tdata}),
	.m_val_axis_tvalid({ADD_13_out_INT_14_tvalid}),
	.m_val_axis_tready({ADD_13_out_INT_14_tready})
);

 assign BB2pred_demux_in1_tdata = BB1pred_demux_out0_1_tdata;
 assign BB2pred_demux_in1_tvalid = BB1pred_demux_out0_1_tvalid;
 assign BB1pred_demux_out0_1_tready = BB2pred_demux_in1_tready;

 assign BB2arg_demux_in1_tdata = ADD_13_out_INT_14_tdata;
 assign BB2arg_demux_in1_tvalid = ADD_13_out_INT_14_tvalid;
 assign ADD_13_out_INT_14_tready = BB2arg_demux_in1_tready;

 assign BB2pred_demux_out0_2_tready = 1;
 assign outport_0_0_tdata = BB2arg_demux_out1_3_tdata;
 assign outport_0_0_tvalid = BB2arg_demux_out1_3_tvalid;
 assign BB2arg_demux_out1_3_tready = outport_0_0_tready;


endmodule
