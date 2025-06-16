module barrier_queue #
(
    parameter DATA_WIDTH = 16,
    parameter KEEP_ENABLE = 1,
    parameter KEEP_WIDTH = (DATA_WIDTH/8),
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,

    input wire [DATA_WIDTH-1:0]          s_in_tdata,
    input wire [KEEP_WIDTH-1:0]         s_in_tkeep,
    input wire                          s_in_tlast,
    input wire                          s_in_tvalid,
    output reg                          s_in_tready,

    input wire                          barrier,
    
    output wire  [DATA_WIDTH-1:0]          m_out_tdata,
    output wire  [KEEP_WIDTH-1:0]          m_out_tkeep,
    output wire                            m_out_tlast,
    output wire                            m_out_tvalid,
    input  wire                            m_out_tready
);

logic [DATA_WIDTH-1:0]             fifo_axis_out_tdata;
logic [KEEP_WIDTH-1:0]             fifo_axis_out_tkeep;
logic                              fifo_axis_out_tvalid;
logic                              fifo_axis_out_tlast;
logic                              fifo_axis_out_tready;

localparam FIFO_FRAME_SIZE = KEEP_ENABLE ? (FIFO_SIZE * KEEP_WIDTH) : FIFO_SIZE;
axis_fifo #(
    .DEPTH(FIFO_FRAME_SIZE),
    .DATA_WIDTH(DATA_WIDTH),
    .KEEP_ENABLE(KEEP_ENABLE),
    .KEEP_WIDTH(KEEP_WIDTH),
    .LAST_ENABLE(KEEP_ENABLE),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_in_tdata),
    .s_axis_tkeep(s_in_tkeep),
    .s_axis_tlast(s_in_tlast),
    .s_axis_tvalid(s_in_tvalid),
    .s_axis_tready(s_in_tready),

    // AXI output
    .m_axis_tdata(fifo_axis_out_tdata),
    .m_axis_tkeep(fifo_axis_out_tkeep),
    .m_axis_tlast(fifo_axis_out_tlast),
    .m_axis_tvalid(fifo_axis_out_tvalid),
    .m_axis_tready(fifo_axis_out_tready)
);

assign m_out_tdata = fifo_axis_out_tdata;
assign m_out_tkeep = fifo_axis_out_tkeep;
assign m_out_tlast = fifo_axis_out_tlast;
assign m_out_tvalid = fifo_axis_out_tvalid && barrier;
assign fifo_axis_out_tready = m_out_tready && barrier;

endmodule