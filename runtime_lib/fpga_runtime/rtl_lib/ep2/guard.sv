module guard #
(
    parameter DATA_WIDTH = 16,
    parameter KEEP_WIDTH = (DATA_WIDTH/8),
    parameter IF_STREAM = 1,
    parameter FIFO_SIZE = 16
)
(
    input  wire                       clk,
    input  wire                       rst,

    input wire [DATA_WIDTH-1:0]                    s_guard_axis_tdata,
    input wire [KEEP_WIDTH-1:0]                    s_guard_axis_tkeep,
    input wire                                     s_guard_axis_tlast,
    input wire                                     s_guard_axis_tvalid,
    output reg                                     s_guard_axis_tready,

    input wire                              s_guard_cond_tdata,
    input wire                              s_guard_cond_tvalid,
    output reg                              s_guard_cond_tready,
    
    output reg  [DATA_WIDTH-1:0]             m_guard_axis_tdata,
    output reg  [KEEP_WIDTH-1:0]             m_guard_axis_tkeep,
    output reg                               m_guard_axis_tlast,
    output reg                               m_guard_axis_tvalid,
    input  wire                               m_guard_axis_tready
);
localparam FILTER_START = 0;
localparam FILTERING = 1;
reg [1:0] state;
reg cond_reg;

localparam FIFO_FRAME_SIZE = IF_STREAM ? (FIFO_SIZE * KEEP_WIDTH) : FIFO_SIZE;
wire [KEEP_WIDTH-1:0]    wrapped_s_guard_axis_tkeep;
wire                     wrapped_s_guard_axis_tlast;

assign wrapped_s_guard_axis_tkeep = IF_STREAM ? s_guard_axis_tkeep : ((1<<KEEP_WIDTH) - 1);
assign wrapped_s_guard_axis_tlast = IF_STREAM ? s_guard_axis_tlast : 1;

always @(posedge clk) begin
    if(rst) begin
        state <= FILTER_START;
    end
    else begin
        if(state == FILTER_START && s_guard_cond_tvalid && s_guard_cond_tready) begin
            if(s_guard_axis_tvalid && s_guard_axis_tready) begin
                if(!wrapped_s_guard_axis_tlast) begin
                    state <= FILTERING;
                    cond_reg <= s_guard_cond_tdata;
                end
            end
        end
        else if(state == FILTERING) begin
            if(s_guard_axis_tvalid && s_guard_axis_tready && wrapped_s_guard_axis_tlast) begin
                state <= FILTER_START;
            end
        end
    end
end

always @* begin
    s_guard_axis_tready = 0;
    s_guard_cond_tready = 0;
    m_guard_axis_tdata = s_guard_axis_tdata;
    m_guard_axis_tkeep = wrapped_s_guard_axis_tkeep;
    m_guard_axis_tlast = wrapped_s_guard_axis_tlast;
    m_guard_axis_tvalid = 0;
    if(state == FILTER_START && s_guard_cond_tvalid && s_guard_axis_tvalid && m_guard_axis_tready) begin
        s_guard_axis_tready = 1;
        s_guard_cond_tready = 1;
        m_guard_axis_tvalid = s_guard_cond_tdata;
    end
    else if(state == FILTERING && s_guard_axis_tvalid && m_guard_axis_tready) begin
        s_guard_axis_tready = 1;
        m_guard_axis_tvalid = cond_reg;
    end
end


endmodule
