module ctrl_mux #
(
    parameter DATA_WIDTH = 16,
    parameter S_COUNT = 2,
    parameter SELECTOR_WIDTH = $clog2(S_COUNT),
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

    output wire  [DATA_WIDTH-1:0]             m_val_axis_tdata,
    output wire  [KEEP_WIDTH-1:0]             m_val_axis_tkeep,
    output wire                               m_val_axis_tlast,
    output wire                               m_val_axis_tvalid,
    input  wire                               m_val_axis_tready,

    input wire  [SELECTOR_WIDTH-1:0]        s_selector_tdata,
    input wire                              s_selector_tvalid,
    output wire                             s_selector_tready
);

localparam FIFO_FRAME_SIZE = KEEP_ENABLE ? (FIFO_SIZE * KEEP_WIDTH) : FIFO_SIZE;


reg   [S_COUNT*DATA_WIDTH-1:0]             grouped_fifo_axis_out_tdata;
reg   [S_COUNT*KEEP_WIDTH-1:0]             grouped_fifo_axis_out_tkeep;
reg   [S_COUNT-1:0]                        grouped_fifo_axis_out_tvalid;
reg   [S_COUNT-1:0]                        grouped_fifo_axis_out_tlast;
wire  [S_COUNT-1:0]                        grouped_fifo_axis_out_tready;

reg   [10-1:0]                        debug_counter_in [S_COUNT-1:0];
reg   [10-1:0]                        debug_counter_out;

always @(posedge clk) begin
    if (rst) begin
        debug_counter_out <= 0;
    end else
    if (m_val_axis_tvalid && m_val_axis_tready) begin
        debug_counter_out <= debug_counter_out + 1;
    end
end

genvar  rid;
generate
    for (rid=0; rid<S_COUNT; rid = rid + 1) begin: inport
        wire [KEEP_WIDTH-1:0]    wrapped_s_val_axis_tkeep;
        wire                     wrapped_s_val_axis_tlast;

        assign wrapped_s_val_axis_tkeep = KEEP_ENABLE ? s_val_axis_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH] : ((1<<KEEP_WIDTH) - 1);
        assign wrapped_s_val_axis_tlast = KEEP_ENABLE ? s_val_axis_tlast[rid] : 1;


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
        always @(posedge clk) begin
            if (rst) begin
                debug_counter_in[rid] <= 0;
            end 
            else if (fifo_axis_in_tvalid && fifo_axis_in_tready) begin
                debug_counter_in[rid] <= debug_counter_in[rid] + 1;
            end
        end

        always @* begin
            // FIFO in wires
            fifo_axis_in_tdata = s_val_axis_tdata[rid*(DATA_WIDTH) +: DATA_WIDTH];
            fifo_axis_in_tkeep = wrapped_s_val_axis_tkeep;
            fifo_axis_in_tlast = wrapped_s_val_axis_tlast;
            fifo_axis_in_tvalid = s_val_axis_tvalid[rid];
            s_val_axis_tready[rid] = fifo_axis_in_tready;

            // FIFO out wires
            grouped_fifo_axis_out_tdata[rid*(DATA_WIDTH) +: DATA_WIDTH] = fifo_axis_out_tdata;
            grouped_fifo_axis_out_tkeep[rid*(KEEP_WIDTH) +: KEEP_WIDTH] = fifo_axis_out_tkeep;
            grouped_fifo_axis_out_tlast[rid] = fifo_axis_out_tlast;
            grouped_fifo_axis_out_tvalid[rid] = fifo_axis_out_tvalid;
            fifo_axis_out_tready =grouped_fifo_axis_out_tready[rid];
        end
    end
endgenerate

wire if_last = KEEP_ENABLE ? m_val_axis_tlast : 1'b1;
wire if_any_enter = |(grouped_fifo_axis_out_tvalid & grouped_fifo_axis_out_tready);
wire if_entered_is_last = | (grouped_fifo_axis_out_tvalid & grouped_fifo_axis_out_tready & grouped_fifo_axis_out_tlast);
assign s_selector_tready = if_any_enter && if_entered_is_last;

axis_mux #(
    .S_COUNT(S_COUNT),
    .DATA_WIDTH(DATA_WIDTH),
    .KEEP_ENABLE(KEEP_ENABLE),
    .KEEP_WIDTH(KEEP_WIDTH),
    .USER_ENABLE(0)
)
axis_mux (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(grouped_fifo_axis_out_tdata),
    .s_axis_tvalid(grouped_fifo_axis_out_tvalid),
    .s_axis_tready(grouped_fifo_axis_out_tready),
    .s_axis_tkeep(grouped_fifo_axis_out_tkeep),
    .s_axis_tlast(grouped_fifo_axis_out_tlast),

    // AXI output
    .m_axis_tdata(m_val_axis_tdata),
    .m_axis_tvalid(m_val_axis_tvalid),
    .m_axis_tready(m_val_axis_tready),
    .m_axis_tkeep(m_val_axis_tkeep),
    .m_axis_tlast(m_val_axis_tlast),

    // Control
    .enable(s_selector_tvalid),
    .select(s_selector_tdata)
);


endmodule