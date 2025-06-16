module ctrl_demux #
(
    parameter DATA_WIDTH = 16,
    parameter D_COUNT = 2,
    parameter SELECTOR_WIDTH = $clog2(D_COUNT),
    parameter KEEP_ENABLE = 1, 
    parameter USER_ENABLE = 0,
    parameter KEEP_WIDTH = (DATA_WIDTH/8),
    parameter FIFO_SIZE = 32
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [DATA_WIDTH-1:0]         s_val_axis_tdata,
    input wire [KEEP_WIDTH-1:0]         s_val_axis_tkeep,
    input wire                          s_val_axis_tlast,
    input wire                          s_val_axis_tvalid,
    output reg                          s_val_axis_tready,

    output reg  [D_COUNT*DATA_WIDTH-1:0]             m_val_axis_tdata,
    output reg  [D_COUNT*KEEP_WIDTH-1:0]             m_val_axis_tkeep,
    output reg  [D_COUNT-1:0]                        m_val_axis_tlast,
    output reg  [D_COUNT-1:0]                        m_val_axis_tvalid,
    input  wire  [D_COUNT-1:0]                        m_val_axis_tready,

    input wire  [SELECTOR_WIDTH-1:0]        s_dispatcher_tdata,
    input wire                              s_dispatcher_tvalid,
    output wire                             s_dispatcher_tready
);


localparam FIFO_FRAME_SIZE = KEEP_ENABLE ? (FIFO_SIZE * KEEP_WIDTH) : FIFO_SIZE;
wire [KEEP_WIDTH-1:0]    wrapped_s_val_axis_tkeep;
wire                     wrapped_s_val_axis_tlast;

assign wrapped_s_val_axis_tkeep = KEEP_ENABLE ? s_val_axis_tkeep : ((1<<KEEP_WIDTH) - 1);
assign wrapped_s_val_axis_tlast = KEEP_ENABLE ? s_val_axis_tlast : 1;

wire [DATA_WIDTH-1:0]             in_fifo_axis_out_tdata;
wire [KEEP_WIDTH-1:0]             in_fifo_axis_out_tkeep;
wire                              in_fifo_axis_out_tvalid;
wire                              in_fifo_axis_out_tlast;
reg                               in_fifo_axis_out_tready;
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
in_fifo (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(s_val_axis_tdata),
    .s_axis_tkeep(wrapped_s_val_axis_tkeep),
    .s_axis_tlast(wrapped_s_val_axis_tlast),
    .s_axis_tvalid(s_val_axis_tvalid),
    .s_axis_tready(s_val_axis_tready),

    // AXI output
    .m_axis_tdata(in_fifo_axis_out_tdata),
    .m_axis_tkeep(in_fifo_axis_out_tkeep),
    .m_axis_tlast(in_fifo_axis_out_tlast),
    .m_axis_tvalid(in_fifo_axis_out_tvalid),
    .m_axis_tready(in_fifo_axis_out_tready)
);

wire   [D_COUNT*DATA_WIDTH-1:0]             grouped_fifo_axis_out_tdata;
wire   [D_COUNT*KEEP_WIDTH-1:0]             grouped_fifo_axis_out_tkeep;
wire   [D_COUNT-1:0]                        grouped_fifo_axis_out_tvalid;
wire   [D_COUNT-1:0]                        grouped_fifo_axis_out_tlast;
reg  [D_COUNT-1:0]                        grouped_fifo_axis_out_tready;

assign s_dispatcher_tready = if_first_segment && in_fifo_axis_out_tvalid && in_fifo_axis_out_tready;

reg if_first_segment;
always @(posedge clk) begin
    if (rst) begin
        if_first_segment <= 1'b1;
    end else begin
        if(in_fifo_axis_out_tvalid && in_fifo_axis_out_tready) begin
            if_first_segment <= 1'b0;
            if(in_fifo_axis_out_tlast) begin
                if_first_segment <= 1'b1;
            end
        end
    end
end

wire enbale;
assign enable = if_first_segment ? s_dispatcher_tvalid : 1'b1;

axis_demux #(
    .M_COUNT(D_COUNT),
    .DATA_WIDTH(DATA_WIDTH),
    .KEEP_ENABLE(KEEP_ENABLE),
    .KEEP_WIDTH(KEEP_WIDTH),
    .USER_ENABLE(0)
)
axis_demux (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(in_fifo_axis_out_tdata),
    .s_axis_tvalid(in_fifo_axis_out_tvalid),
    .s_axis_tready(in_fifo_axis_out_tready),
    .s_axis_tkeep(in_fifo_axis_out_tkeep),
    .s_axis_tlast(in_fifo_axis_out_tlast),

    // AXI output
    .m_axis_tdata(grouped_fifo_axis_out_tdata),
    .m_axis_tvalid(grouped_fifo_axis_out_tvalid ),
    .m_axis_tready(grouped_fifo_axis_out_tready),
    .m_axis_tkeep(grouped_fifo_axis_out_tkeep),
    .m_axis_tlast(grouped_fifo_axis_out_tlast),

    // Control
    .enable(enable),
    .drop(0),
    .select(s_dispatcher_tdata)
);

genvar  rid;
generate
    for (rid=0; rid<D_COUNT; rid = rid + 1) begin: inport


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
            .KEEP_ENABLE(KEEP_ENABLE),
            .KEEP_WIDTH(KEEP_WIDTH),
            .LAST_ENABLE(KEEP_ENABLE),
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
            // FIFO in wires
            fifo_axis_in_tdata = grouped_fifo_axis_out_tdata[rid*(DATA_WIDTH) +: DATA_WIDTH];
            fifo_axis_in_tkeep = grouped_fifo_axis_out_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH];
            fifo_axis_in_tlast = grouped_fifo_axis_out_tlast[rid];
            fifo_axis_in_tvalid = grouped_fifo_axis_out_tvalid[rid];
            grouped_fifo_axis_out_tready[rid] = fifo_axis_in_tready;

            // FIFO out wires
            m_val_axis_tdata[rid*(DATA_WIDTH) +: DATA_WIDTH] = fifo_axis_out_tdata;
            m_val_axis_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH] = fifo_axis_out_tkeep;
            m_val_axis_tlast[rid] = fifo_axis_out_tlast;
            m_val_axis_tvalid[rid] = fifo_axis_out_tvalid;
            fifo_axis_out_tready = m_val_axis_tready[rid];
        end
    end
endgenerate




endmodule