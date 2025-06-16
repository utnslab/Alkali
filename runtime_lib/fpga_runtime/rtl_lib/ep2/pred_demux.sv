module pred_demux #
(
    parameter VAL_WIDTH = 1,
    parameter PORT_COUNT = 2,
    parameter IF_STREAM = 0,
    parameter IF_LOCAL_PRED_OUT = 1, // can be 0 or 1
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [PORT_COUNT*VAL_WIDTH-1:0]         s_demux_in_tdata,
    input wire [PORT_COUNT-1:0]                    s_demux_in_tvalid,
    output reg [PORT_COUNT-1:0]                    s_demux_in_tready,

    output wire  [PORT_COUNT-1:0]             m_pred_out_tdata,
    output wire                              m_pred_out_tvalid,
    input  wire                              m_pred_out_tready,

    output wire  [VAL_WIDTH-1:0]             m_demux_out_tdata,
    output wire                              m_demux_out_tvalid,
    input  wire                              m_demux_out_tready
);


reg [PORT_COUNT-1:0]    pred_data_bitmask;
reg [PORT_COUNT-1:0]    pred_valid_bitmask;
wire                    warpped_pred_out_tready;

assign warpped_pred_out_tready = (IF_LOCAL_PRED_OUT > 0) ? m_pred_out_tready : 1;

assign m_pred_out_tdata = pred_data_bitmask;
assign m_pred_out_tvalid = &pred_valid_bitmask;

assign m_demux_out_tdata = &pred_data_bitmask;
assign m_demux_out_tvalid = &pred_valid_bitmask;

genvar  rid;
generate
    for (rid=0; rid<PORT_COUNT; rid = rid + 1) begin: inport
        reg [VAL_WIDTH-1:0]              fifo_axis_in_tdata;
        reg                              fifo_axis_in_tvalid;
        wire                             fifo_axis_in_tready;
        

        wire [VAL_WIDTH-1:0]             fifo_axis_out_tdata;
        wire                             fifo_axis_out_tvalid;
        reg                              fifo_axis_out_tready;

        axis_fifo #(
            .DEPTH(FIFO_SIZE),
            .DATA_WIDTH(VAL_WIDTH),
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
            .s_axis_tdata(fifo_axis_in_tdata),
            .s_axis_tvalid(fifo_axis_in_tvalid),
            .s_axis_tready(fifo_axis_in_tready),

            // AXI output
            .m_axis_tdata(fifo_axis_out_tdata),
            .m_axis_tvalid(fifo_axis_out_tvalid),
            .m_axis_tready(fifo_axis_out_tready)
        );

        always @* begin
            pred_data_bitmask[rid] = fifo_axis_out_tdata;
            pred_valid_bitmask[rid] = fifo_axis_out_tvalid;
            fifo_axis_out_tready = m_demux_out_tready && warpped_pred_out_tready && (&pred_valid_bitmask);
            

            s_demux_in_tready[rid] = fifo_axis_in_tready;
            fifo_axis_in_tdata = s_demux_in_tdata[rid*(VAL_WIDTH) +: VAL_WIDTH];
            fifo_axis_in_tvalid = s_demux_in_tvalid[rid];
        end
    end
endgenerate



endmodule