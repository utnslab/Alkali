module ctrl_all_to_all #
(
    parameter DATA_WIDTH = 16,
    parameter S_COUNT = 2,
    parameter D_COUNT = 2,
    parameter SELECTOR_WIDTH = $clog2(S_COUNT),
    parameter DISPATCHER_WIDTH = $clog2(D_COUNT),
    parameter KEEP_ENABLE = 1, 
    parameter USER_ENABLE = 0,
    parameter KEEP_WIDTH = (DATA_WIDTH/8),
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [S_COUNT*DATA_WIDTH-1:0]         s_val_axis_tdata,
    input wire [S_COUNT*KEEP_WIDTH-1:0]         s_val_axis_tkeep,
    input wire [S_COUNT-1:0]                    s_val_axis_tlast,
    input wire [S_COUNT-1:0]                    s_val_axis_tvalid,
    output reg [S_COUNT-1:0]                    s_val_axis_tready,

    input wire  [SELECTOR_WIDTH-1:0]        s_selector_tdata,
    input wire                              s_selector_tvalid,
    output wire                             s_selector_tready,


    output reg  [D_COUNT*DATA_WIDTH-1:0]             m_val_axis_tdata,
    output reg  [D_COUNT*KEEP_WIDTH-1:0]             m_val_axis_tkeep,
    output reg  [D_COUNT-1:0]                        m_val_axis_tlast,
    output reg  [D_COUNT-1:0]                        m_val_axis_tvalid,
    input  wire  [D_COUNT-1:0]                       m_val_axis_tready,

    input wire  [DISPATCHER_WIDTH-1:0]        s_dispatcher_tdata,
    input wire                                s_dispatcher_tvalid,
    output wire                               s_dispatcher_tready
);


wire [DATA_WIDTH-1:0]         tmp_val_axis_tdata;
wire [KEEP_WIDTH-1:0]         tmp_val_axis_tkeep;
wire                          tmp_val_axis_tlast;
wire                          tmp_val_axis_tvalid;
wire                          tmp_val_axis_tready;

ctrl_mux#(
.S_COUNT  (S_COUNT),
.DATA_WIDTH(DATA_WIDTH),
.KEEP_ENABLE (KEEP_ENABLE)
)ctrl_mux(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata(s_val_axis_tdata),
	.s_val_axis_tkeep(s_val_axis_tkeep),
	.s_val_axis_tlast(s_val_axis_tlast),
	.s_val_axis_tvalid(s_val_axis_tvalid),
	.s_val_axis_tready(s_val_axis_tready),
	//(des)mux out
	.m_val_axis_tdata(tmp_val_axis_tdata),
	.m_val_axis_tkeep(tmp_val_axis_tkeep),
	.m_val_axis_tlast(tmp_val_axis_tlast),
	.m_val_axis_tvalid(tmp_val_axis_tvalid),
	.m_val_axis_tready(tmp_val_axis_tready),
	//selector wire
	.s_selector_tdata(s_selector_tdata),
	.s_selector_tvalid(s_selector_tvalid),
	.s_selector_tready(s_selector_tready)
);

ctrl_demux#(
.D_COUNT  (D_COUNT),
.DATA_WIDTH(DATA_WIDTH),
.KEEP_ENABLE (KEEP_ENABLE)
)ctrl_demux_177(
	 .clk(clk), 
	 .rst(rst) ,
	//(de)mux in
	.s_val_axis_tdata({tmp_val_axis_tdata}),
	.s_val_axis_tkeep({tmp_val_axis_tkeep}),
	.s_val_axis_tlast({tmp_val_axis_tlast}),
	.s_val_axis_tvalid({tmp_val_axis_tvalid}),
	.s_val_axis_tready({tmp_val_axis_tready}),
	//(des)mux out
	.m_val_axis_tdata(m_val_axis_tdata),
	.m_val_axis_tkeep(m_val_axis_tkeep),
	.m_val_axis_tlast(m_val_axis_tlast),
	.m_val_axis_tvalid(m_val_axis_tvalid),
	.m_val_axis_tready(m_val_axis_tready),
	//selector wire
	.s_dispatcher_tdata(s_dispatcher_tdata),
	.s_dispatcher_tvalid(s_dispatcher_tvalid),
	.s_dispatcher_tready(s_dispatcher_tready)
);
endmodule