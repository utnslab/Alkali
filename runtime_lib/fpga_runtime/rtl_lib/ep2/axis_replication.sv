module axis_replication #
(
    parameter DATA_WIDTH = 16,
    parameter KEEP_WIDTH = IF_STREAM ? (DATA_WIDTH/8) : 1,
    parameter IF_STREAM = 1,
    parameter REAPLICA_COUNT = 2,
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [DATA_WIDTH-1:0]             s_axis_in_tdata,
    input wire [KEEP_WIDTH-1:0]             s_axis_in_tkeep,
    input wire                              s_axis_in_tvalid,
    input wire                              s_axis_in_tlast,
    output reg                              s_axis_in_tready,


    output reg [REAPLICA_COUNT*DATA_WIDTH-1:0]             m_axis_out_tdata,
    output reg [REAPLICA_COUNT*KEEP_WIDTH-1:0]             m_axis_out_tkeep,
    output reg [REAPLICA_COUNT-1:0]                        m_axis_out_tvalid,
    input  wire [REAPLICA_COUNT-1:0]                       m_axis_out_tready,
    output reg  [REAPLICA_COUNT-1:0]                       m_axis_out_tlast
);

localparam FIFO_FRAME_SIZE = IF_STREAM ? (FIFO_SIZE * KEEP_WIDTH) : FIFO_SIZE;
wire [KEEP_WIDTH-1:0]    wrapped_s_axis_in_tkeep;
wire                     wrapped_s_axis_in_tlast;

assign wrapped_s_axis_in_tkeep = IF_STREAM ? s_axis_in_tkeep : ((1<<KEEP_WIDTH) - 1);
assign wrapped_s_axis_in_tlast = IF_STREAM ? s_axis_in_tlast : 1;


reg [REAPLICA_COUNT-1:0]    fifo_in_array_ready;
assign s_axis_in_tready = &fifo_in_array_ready;

genvar  rid;
generate
    for (rid=0; rid<REAPLICA_COUNT; rid = rid + 1) begin: replicate
        reg [DATA_WIDTH-1:0]              fifo_axis_in_tdata;
        reg [KEEP_WIDTH-1:0]              fifo_axis_in_tkeep;
        reg                               fifo_axis_in_tvalid;
        reg                               fifo_axis_in_tlast;
        wire                              fifo_axis_in_tready;
        

        wire [DATA_WIDTH-1:0]             fifo_axis_out_tdata;
        wire [KEEP_WIDTH-1:0]             fifo_axis_out_tkeep;
        wire                              fifo_axis_out_tvalid;
        wire                              fifo_axis_out_tlast;
        reg                               fifo_axis_out_tready;

        axis_fifo #(
            .DEPTH(FIFO_FRAME_SIZE),
            .DATA_WIDTH(DATA_WIDTH),
            .KEEP_ENABLE(IF_STREAM),
            .KEEP_WIDTH(KEEP_WIDTH),
            .LAST_ENABLE(IF_STREAM),
            .ID_ENABLE(0),
            .DEST_ENABLE(0),
            .USER_ENABLE(0),
            .FRAME_FIFO(0)
        )
        in_fifo (
            .clk(clk),
            .rst(rst),
            // AXI input
            .s_axis_tdata(fifo_axis_in_tdata),
            .s_axis_tkeep(fifo_axis_in_tkeep),
            .s_axis_tlast(fifo_axis_in_tlast),
            .s_axis_tvalid(fifo_axis_in_tvalid),
            .s_axis_tready(fifo_axis_in_tready),

            // AXI output
            .m_axis_tdata(fifo_axis_out_tdata),
            .m_axis_tkeep(fifo_axis_out_tkeep),
            .m_axis_tlast(fifo_axis_out_tlast),
            .m_axis_tvalid(fifo_axis_out_tvalid),
            .m_axis_tready(fifo_axis_out_tready)
        );

        always @* begin
            m_axis_out_tdata[rid*(DATA_WIDTH) +: DATA_WIDTH] = fifo_axis_out_tdata;
            // m_axis_out_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH] = 0;
            m_axis_out_tlast[rid] = fifo_axis_out_tlast;
            m_axis_out_tvalid[rid] = fifo_axis_out_tvalid;
            fifo_axis_out_tready = m_axis_out_tready[rid];

            fifo_in_array_ready[rid] = fifo_axis_in_tready;
            fifo_axis_in_tdata = s_axis_in_tdata;
            fifo_axis_in_tkeep = wrapped_s_axis_in_tkeep;
            fifo_axis_in_tlast = wrapped_s_axis_in_tlast;
            fifo_axis_in_tvalid = s_axis_in_tvalid && s_axis_in_tready;
        end
    end


endgenerate
endmodule