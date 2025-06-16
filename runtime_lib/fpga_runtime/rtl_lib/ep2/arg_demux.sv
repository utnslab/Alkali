module arg_demux #
(
    parameter VAL_WIDTH = 16,
    parameter KEEP_WIDTH = (VAL_WIDTH/8),
    parameter IF_STREAM = 1,
    parameter PORT_COUNT = 2,
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,

    input wire [PORT_COUNT*VAL_WIDTH-1:0]          s_demux_in_tdata,
    input wire [PORT_COUNT*KEEP_WIDTH-1:0]         s_demux_in_tkeep,
    input wire [PORT_COUNT-1:0]                    s_demux_in_tlast,
    input wire [PORT_COUNT-1:0]                    s_demux_in_tvalid,
    output reg [PORT_COUNT-1:0]                    s_demux_in_tready,

    input wire  [PORT_COUNT-1:0]             s_pred_in_tdata,
    input wire                              s_pred_in_tvalid,
    output reg                              s_pred_in_tready,
    
    output wire  [VAL_WIDTH-1:0]              m_demux_out_tdata,
    output wire  [KEEP_WIDTH-1:0]             m_demux_out_tkeep,
    output wire                               m_demux_out_tlast,
    output wire                               m_demux_out_tvalid,
    input  wire                               m_demux_out_tready
);

localparam FIFO_FRAME_SIZE = IF_STREAM ? (FIFO_SIZE * KEEP_WIDTH) : FIFO_SIZE;

reg [PORT_COUNT-1:0]    arg_valid_bitmask;

reg [PORT_COUNT*VAL_WIDTH-1:0]              grouped_fifo_axis_out_tdata;
reg [PORT_COUNT*KEEP_WIDTH-1:0]             grouped_fifo_axis_out_tkeep;
reg [PORT_COUNT-1:0]                        grouped_fifo_axis_out_tvalid;
reg [PORT_COUNT-1:0]                        grouped_fifo_axis_out_tlast;
wire  [PORT_COUNT-1:0]                        grouped_fifo_axis_out_tready;

assign s_pred_in_tready = |(grouped_fifo_axis_out_tvalid & grouped_fifo_axis_out_tready);

genvar  rid;
generate
    for (rid=0; rid<PORT_COUNT; rid = rid + 1) begin: inport
        wire [KEEP_WIDTH-1:0]    wrapped_s_demux_in_tkeep;
        wire                     wrapped_s_demux_in_tlast;

        assign wrapped_s_demux_in_tkeep = IF_STREAM ? s_demux_in_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH] : ((1<<KEEP_WIDTH) - 1);
        assign wrapped_s_demux_in_tlast = IF_STREAM ? s_demux_in_tlast[rid] : 1;


        reg [VAL_WIDTH-1:0]              fifo_axis_in_tdata;
        reg [KEEP_WIDTH-1:0]              fifo_axis_in_tkeep;
        reg                               fifo_axis_in_tvalid;
        reg                               fifo_axis_in_tlast;
        wire                              fifo_axis_in_tready;
        
        wire [VAL_WIDTH-1:0]             fifo_axis_out_tdata;
        wire [KEEP_WIDTH-1:0]             fifo_axis_out_tkeep;
        wire                              fifo_axis_out_tvalid;
        wire                              fifo_axis_out_tlast;
        reg                               fifo_axis_out_tready;

        reg selected;
        reg everyone_is_valid;

        axis_fifo #(
            .DEPTH(FIFO_FRAME_SIZE),
            .DATA_WIDTH(VAL_WIDTH),
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
            selected = s_pred_in_tdata[rid] == 1'b1;
            everyone_is_valid = (& arg_valid_bitmask) && s_pred_in_tvalid;
            
            // FIFO in wires
            fifo_axis_in_tdata = s_demux_in_tdata[rid*(VAL_WIDTH) +: VAL_WIDTH];
            fifo_axis_in_tkeep = wrapped_s_demux_in_tkeep;
            fifo_axis_in_tlast = wrapped_s_demux_in_tlast;
            fifo_axis_in_tvalid = s_demux_in_tvalid[rid];
            s_demux_in_tready[rid] = fifo_axis_in_tready;

            // FIFO out wires
            grouped_fifo_axis_out_tdata[rid*(VAL_WIDTH) +: VAL_WIDTH] = fifo_axis_out_tdata;
            grouped_fifo_axis_out_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH] = fifo_axis_out_tkeep;
            grouped_fifo_axis_out_tlast[rid] = fifo_axis_out_tlast;
            
            arg_valid_bitmask[rid] = fifo_axis_out_tvalid;
            fifo_axis_out_tready = everyone_is_valid &&
                            (selected ? grouped_fifo_axis_out_tready[rid] : (| grouped_fifo_axis_out_tready));
            grouped_fifo_axis_out_tvalid[rid] = everyone_is_valid && selected;
        end
    end
endgenerate

axis_arb_mux #(
    .S_COUNT(PORT_COUNT),
    .DATA_WIDTH(VAL_WIDTH),
    .KEEP_ENABLE(IF_STREAM),
    .KEEP_WIDTH(KEEP_WIDTH),
    .USER_ENABLE(0),
    .ARB_TYPE("ROUND_ROBIN")
)
lookup_mux (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(grouped_fifo_axis_out_tdata),
    .s_axis_tvalid(grouped_fifo_axis_out_tvalid),
    .s_axis_tready(grouped_fifo_axis_out_tready),
    .s_axis_tkeep(grouped_fifo_axis_out_tkeep),
    .s_axis_tlast(grouped_fifo_axis_out_tlast),

    // AXI output
    .m_axis_tdata(m_demux_out_tdata),
    .m_axis_tvalid(m_demux_out_tvalid),
    .m_axis_tready(m_demux_out_tready),
    .m_axis_tkeep(m_demux_out_tkeep),
    .m_axis_tlast(m_demux_out_tlast)
);


endmodule
