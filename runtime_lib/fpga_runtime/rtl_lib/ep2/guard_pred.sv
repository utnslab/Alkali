module guard_pred#(
    parameter COND_WIDTH = 2,
    parameter GROUND_TRUTH = 3,
    parameter REPLICATED_OUT_NUM = 2,
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [COND_WIDTH-1:0]         s_guard_cond_tdata,
    input wire [COND_WIDTH-1:0]         s_guard_cond_tvalid,
    output reg [COND_WIDTH-1:0]         s_guard_cond_tready,

    output wire  [REPLICATED_OUT_NUM-1:0]            m_guard_cond_tdata,
    output wire  [REPLICATED_OUT_NUM-1:0]            m_guard_cond_tvalid,
    input  wire  [REPLICATED_OUT_NUM-1:0]            m_guard_cond_tready
);

reg [COND_WIDTH-1:0] ground_truth_reg = GROUND_TRUTH;
reg [COND_WIDTH-1:0] guard_fifo_valid_array;
reg [COND_WIDTH-1:0] guard_fifo_data_array;


genvar  rid;
generate
    for (rid=0; rid<COND_WIDTH; rid = rid + 1) begin: inport
        reg                               fifo_cond_in_tdata;
        reg                               fifo_cond_in_tvalid;
        wire                              fifo_cond_in_tready;
        
        wire                              fifo_cond_out_tdata;
        wire                              fifo_cond_out_tvalid;
        reg                               fifo_cond_out_tready;

        axis_fifo #(
            .DEPTH(FIFO_SIZE),
            .DATA_WIDTH(1),
            .KEEP_ENABLE(0),
            .LAST_ENABLE(0),
            .ID_ENABLE(0),
            .DEST_ENABLE(0),
            .USER_ENABLE(0),
            .FRAME_FIFO(0)
        )
        in_fifo (
            .clk(clk),
            .rst(rst),
            // AXI input
            .s_axis_tdata(fifo_cond_in_tdata),
            .s_axis_tvalid(fifo_cond_in_tvalid),
            .s_axis_tready(fifo_cond_in_tready),

            // AXI output
            .m_axis_tdata(fifo_cond_out_tdata),
            .m_axis_tvalid(fifo_cond_out_tvalid),
            .m_axis_tready(fifo_cond_out_tready)
        );

        always @(*) begin
            fifo_cond_in_tdata = s_guard_cond_tdata[rid];
            fifo_cond_in_tvalid = s_guard_cond_tvalid[rid];
            s_guard_cond_tready[rid] = fifo_cond_in_tready;

            guard_fifo_valid_array[rid] = fifo_cond_out_tvalid;
            guard_fifo_data_array[rid] = fifo_cond_out_tdata;
            fifo_cond_out_tready = replicate_guard_cond_in_tready && (&guard_fifo_valid_array);
        end
    end
endgenerate

wire            replicate_guard_cond_in_tdata;
wire            replicate_guard_cond_in_tvalid;
wire            replicate_guard_cond_in_tready;

assign replicate_guard_cond_in_tdata = (guard_fifo_data_array == ground_truth_reg);
assign replicate_guard_cond_in_tvalid = (&guard_fifo_valid_array);


axis_replication#(
.DATA_WIDTH(COND_WIDTH),
.IF_STREAM(0),
.REAPLICA_COUNT(REPLICATED_OUT_NUM)
)axis_replication_6(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata(replicate_guard_cond_in_tdata),
	.s_axis_in_tvalid(replicate_guard_cond_in_tvalid),
	.s_axis_in_tready(replicate_guard_cond_in_tready),
	//
	.m_axis_out_tdata(m_guard_cond_tdata),
	.m_axis_out_tvalid(m_guard_cond_tvalid),
	.m_axis_out_tready(m_guard_cond_tready)
);
endmodule