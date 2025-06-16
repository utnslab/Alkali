module select #
(
    parameter VAL_WIDTH = 16,
    parameter COND_WIDTH = 1,
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input  wire [COND_WIDTH-1:0]             s_cond_axis_tdata,
    input  wire                              s_cond_axis_tvalid,
    output wire                              s_cond_axis_tready,

    input  wire [VAL_WIDTH-1:0]              s_true_val_axis_tdata,
    input  wire                              s_true_val_axis_tvalid,
    output wire                              s_true_val_axis_tready,

    input wire [VAL_WIDTH-1:0]              s_false_val_axis_tdata,
    input wire                              s_false_val_axis_tvalid,
    output wire                             s_false_val_axis_tready,

    output reg  [VAL_WIDTH-1:0]          m_val_axis_tdata,
    output reg                           m_val_axis_tvalid,
    input  wire                          m_val_axis_tready
);



wire [COND_WIDTH-1:0]             fifo_cond_axis_tdata;
wire                              fifo_cond_axis_tvalid;
reg                              fifo_cond_axis_tready;

wire [VAL_WIDTH-1:0]              fifo_true_val_axis_tdata;
wire                              fifo_true_val_axis_tvalid;
reg                              fifo_true_val_axis_tready;


wire [VAL_WIDTH-1:0]              fifo_false_val_axis_tdata;
wire                              fifo_false_val_axis_tvalid;
reg                              fifo_false_val_axis_tready;

always@* begin
    m_val_axis_tdata = 0;

    fifo_cond_axis_tready = m_val_axis_tready && fifo_true_val_axis_tvalid && fifo_false_val_axis_tvalid;
    fifo_true_val_axis_tready = m_val_axis_tready && fifo_cond_axis_tvalid && fifo_false_val_axis_tvalid;
    fifo_false_val_axis_tready = m_val_axis_tready && fifo_cond_axis_tvalid && fifo_true_val_axis_tvalid;

    m_val_axis_tvalid = fifo_cond_axis_tvalid && fifo_true_val_axis_tvalid && fifo_false_val_axis_tvalid;
    
    m_val_axis_tdata = fifo_cond_axis_tdata ? fifo_true_val_axis_tdata : fifo_false_val_axis_tdata;
end
axis_fifo #(
    .DEPTH(FIFO_SIZE),
    .DATA_WIDTH(COND_WIDTH),
    .KEEP_ENABLE(0),
    .LAST_ENABLE(0),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
cond_fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_cond_axis_tdata),
    .s_axis_tvalid(s_cond_axis_tvalid),
    .s_axis_tready(s_cond_axis_tready),

    // AXI output
    .m_axis_tdata(fifo_cond_axis_tdata),
    .m_axis_tvalid(fifo_cond_axis_tvalid),
    .m_axis_tready(fifo_cond_axis_tready)
);


axis_fifo #(
    .DEPTH(FIFO_SIZE),
    .DATA_WIDTH(VAL_WIDTH),
    .KEEP_ENABLE(0),
    .LAST_ENABLE(0),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
if_fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_true_val_axis_tdata),
    .s_axis_tvalid(s_true_val_axis_tvalid),
    .s_axis_tready(s_true_val_axis_tready),

    // AXI output
    .m_axis_tdata(fifo_true_val_axis_tdata),
    .m_axis_tvalid(fifo_true_val_axis_tvalid),
    .m_axis_tready(fifo_false_val_axis_tready)
);

axis_fifo #(
    .DEPTH(FIFO_SIZE),
    .DATA_WIDTH(VAL_WIDTH),
    .KEEP_ENABLE(0),
    .LAST_ENABLE(0),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
else_fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_false_val_axis_tdata),
    .s_axis_tvalid(s_false_val_axis_tvalid),
    .s_axis_tready(s_false_val_axis_tready),

    // AXI output
    .m_axis_tdata(fifo_false_val_axis_tdata),
    .m_axis_tvalid(fifo_false_val_axis_tvalid),
    .m_axis_tready(fifo_false_val_axis_tready)
);

endmodule