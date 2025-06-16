module ifelse_demux #
(
    parameter VAL_WIDTH = 16,
    parameter VAL_KEEP_WIDTH = VAL_WIDTH/8,
    parameter COND_WIDTH = 1,
    parameter IF_STREAM = 1,
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input  wire [COND_WIDTH-1:0]             s_cond_axis_tdata,
    input  wire                              s_cond_axis_tvalid,
    output wire                              s_cond_axis_tready,

    input  wire [VAL_WIDTH-1:0]              s_if_axis_tdata,
    input  wire [VAL_KEEP_WIDTH-1:0]         s_if_axis_tkeep,
    input  wire                              s_if_axis_tvalid,
    input  wire                              s_if_axis_tlast,
    output wire                              s_if_axis_tready,

    input wire [VAL_WIDTH-1:0]              s_else_axis_tdata,
    input wire [VAL_KEEP_WIDTH-1:0]         s_else_axis_tkeep,
    input wire                              s_else_axis_tvalid,
    input wire                              s_else_axis_tlast,
    output wire                             s_else_axis_tready,

    output reg  [VAL_WIDTH-1:0]          m_val_axis_tdata,
    output reg  [VAL_KEEP_WIDTH-1:0]     m_val_axis_tkeep,
    output reg                           m_val_axis_tvalid,
    input  wire                          m_val_axis_tready,
    output reg                           m_val_axis_tlast
);


wire [VAL_KEEP_WIDTH-1:0]    wrapped_s_if_axis_tkeep;
wire                         wrapped_s_if_axis_tlast;

wire [VAL_KEEP_WIDTH-1:0]    wrapped_s_else_axis_tkeep;
wire                         wrapped_s_else_axis_tlast;


assign wrapped_s_if_axis_tkeep = IF_STREAM ? s_if_axis_tkeep : ((1<<VAL_KEEP_WIDTH) - 1);
assign wrapped_s_if_axis_tlast = IF_STREAM ? s_if_axis_tlast : 1;

assign wrapped_s_else_axis_tkeep = IF_STREAM ? s_else_axis_tkeep : ((1<<VAL_KEEP_WIDTH) - 1);
assign wrapped_s_else_axis_tlast = IF_STREAM ? s_else_axis_tlast : 1;


wire [COND_WIDTH-1:0]             fifo_cond_axis_tdata;
wire                              fifo_cond_axis_tvalid;
reg                              fifo_cond_axis_tready;


wire [VAL_WIDTH-1:0]              fifo_if_axis_tdata;
wire [VAL_KEEP_WIDTH-1:0]         fifo_if_axis_tkeep;
wire                              fifo_if_axis_tvalid;
wire                              fifo_if_axis_tlast;
reg                              fifo_if_axis_tready;


wire [VAL_WIDTH-1:0]              fifo_else_axis_tdata;
wire [VAL_KEEP_WIDTH-1:0]         fifo_else_axis_tkeep;
wire                              fifo_else_axis_tvalid;
wire                              fifo_else_axis_tlast;
reg                              fifo_else_axis_tready;

always@* begin
    m_val_axis_tdata = 0;
    m_val_axis_tkeep = 0;
    m_val_axis_tlast = 0;

    fifo_cond_axis_tready = m_val_axis_tready && fifo_if_axis_tvalid && fifo_else_axis_tvalid;
    fifo_if_axis_tready = m_val_axis_tready && fifo_cond_axis_tvalid && fifo_else_axis_tvalid;
    fifo_else_axis_tready = m_val_axis_tready && fifo_cond_axis_tvalid && fifo_if_axis_tvalid;

    m_val_axis_tvalid = fifo_cond_axis_tvalid && fifo_if_axis_tvalid && fifo_else_axis_tvalid;

    if(m_val_axis_tready && m_val_axis_tvalid) begin
        m_val_axis_tdata = fifo_cond_axis_tdata ? fifo_if_axis_tdata : fifo_else_axis_tdata;
        m_val_axis_tkeep = fifo_cond_axis_tdata ? fifo_if_axis_tkeep : fifo_else_axis_tkeep;
        m_val_axis_tlast = fifo_cond_axis_tdata ? fifo_if_axis_tlast : fifo_else_axis_tlast;
    end

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
    .DEPTH(FIFO_SIZE * VAL_KEEP_WIDTH),
    .DATA_WIDTH(VAL_WIDTH),
    .KEEP_ENABLE(1),
    .KEEP_WIDTH(VAL_KEEP_WIDTH),
    .LAST_ENABLE(1),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
if_fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_if_axis_tdata),
    .s_axis_tkeep(wrapped_s_if_axis_tkeep),
    .s_axis_tlast(wrapped_s_if_axis_tlast),
    .s_axis_tvalid(s_if_axis_tvalid),
    .s_axis_tready(s_if_axis_tready),

    // AXI output
    .m_axis_tdata(fifo_if_axis_tdata),
    .m_axis_tkeep(fifo_if_axis_tkeep),
    .m_axis_tlast(fifo_if_axis_tlast),
    .m_axis_tvalid(fifo_if_axis_tvalid),
    .m_axis_tready(fifo_else_axis_tready)
);

axis_fifo #(
    .DEPTH(FIFO_SIZE * VAL_KEEP_WIDTH),
    .DATA_WIDTH(VAL_WIDTH),
    .KEEP_ENABLE(1),
    .KEEP_WIDTH(VAL_KEEP_WIDTH),
    .LAST_ENABLE(1),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
else_fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_else_axis_tdata),
    .s_axis_tkeep(wrapped_s_else_axis_tkeep),
    .s_axis_tlast(wrapped_s_else_axis_tlast),
    .s_axis_tvalid(s_else_axis_tvalid),
    .s_axis_tready(s_else_axis_tready),

    // AXI output
    .m_axis_tdata(fifo_else_axis_tdata),
    .m_axis_tkeep(fifo_else_axis_tkeep),
    .m_axis_tlast(fifo_else_axis_tlast),
    .m_axis_tvalid(fifo_else_axis_tvalid),
    .m_axis_tready(fifo_else_axis_tready)
);

endmodule