module extract #
(
    // Width of AXI stream interfaces in bits
    parameter BUF_DATA_WIDTH = 256,
    // AXI stream tkeep signal width (words per cycle)
    parameter BUF_KEEP_WIDTH = (BUF_DATA_WIDTH/8),
    parameter EXTRACTED_STRUCT_WIDTH = 16
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [BUF_DATA_WIDTH-1:0]           s_inbuf_axis_tdata,
    input wire [BUF_KEEP_WIDTH-1:0]           s_inbuf_axis_tkeep,
    input wire                                s_inbuf_axis_tvalid,
    output reg                                s_inbuf_axis_tready,
    input wire                                s_inbuf_axis_tlast,

    // assume < 64B
    output wire [EXTRACTED_STRUCT_WIDTH-1:0]   m_extracted_axis_tdata,
    output wire                                m_extracted_axis_tvalid,
    input  wire                                m_extracted_axis_tready,

    output wire [BUF_DATA_WIDTH-1:0]           m_outbuf_axis_tdata,
    output wire [BUF_KEEP_WIDTH-1:0]           m_outbuf_axis_tkeep,
    output wire                                m_outbuf_axis_tvalid,
    input  wire                                m_outbuf_axis_tready,
    output wire                                m_outbuf_axis_tlast
);

localparam EXTRACT_STATE = 0;
localparam FORWARD_STATE = 1;
reg [1:0] state;

reg [1:0] shift_occupancy_reg;
reg [1:0] shift_valid_reg;
reg [1:0] shift_last_reg;
reg [(BUF_DATA_WIDTH *2 - 1):0] shift_reg;
reg [(BUF_KEEP_WIDTH *2 - 1):0] shift_keep_reg;

wire shift_ready_in_wire;
reg [1:0] shift_occupancy_wire;
reg [1:0] shift_valid_wire;
reg [1:0] shift_last_wire;
reg [(BUF_DATA_WIDTH *2 - 1):0] shift_wire;
reg [(BUF_KEEP_WIDTH *2 - 1):0] shift_keep_wire;

always @(posedge clk) begin
    if(rst) begin
        shift_reg = 0;
        shift_keep_reg = 0;
        shift_valid_reg = 0;
        shift_last_reg = 0;
        shift_occupancy_reg = 0;
    end
    else begin
        if(!shift_occupancy_reg[1] || (reg_outbuf_axis_tready && reg_outbuf_axis_tvalid)) begin
            shift_reg <= (shift_wire << BUF_DATA_WIDTH);
            shift_keep_reg <= (shift_keep_wire << BUF_KEEP_WIDTH);
            shift_valid_reg <= (shift_valid_wire << 1);
            shift_last_reg <= (shift_last_wire << 1);
            shift_occupancy_reg <= (shift_occupancy_wire << 1);
        end
        else begin
            shift_reg <= (shift_wire);
            shift_keep_reg <= (shift_keep_wire);
            shift_valid_reg <= (shift_valid_wire);
            shift_last_reg <= (shift_last_wire);
            shift_occupancy_reg <= (shift_occupancy_wire);
        end
    end
end

always @(posedge clk) begin
    if(rst) begin
        state <= EXTRACT_STATE;
    end
    else begin

        if(state == EXTRACT_STATE && (s_inbuf_axis_tready && s_inbuf_axis_tvalid)) begin
            if(!s_inbuf_axis_tlast) begin
                state <= FORWARD_STATE;
            end
            else begin
                state <= EXTRACT_STATE;
            end
        end
        else if(state == FORWARD_STATE && (s_inbuf_axis_tready && s_inbuf_axis_tvalid && s_inbuf_axis_tlast)) begin
            state <= EXTRACT_STATE;
        end
    end
end

assign shift_ready_in_wire = !shift_occupancy_reg[0];
always @* begin
    if(state == EXTRACT_STATE) begin
        s_inbuf_axis_tready = reg_extracted_axis_tready && shift_ready_in_wire;
    end
    else begin
        s_inbuf_axis_tready = shift_ready_in_wire;
    end

    reg_extracted_axis_tvalid = (state == EXTRACT_STATE && s_inbuf_axis_tvalid);
    reg_extracted_axis_tdata = 0;

    shift_wire = shift_reg;
    shift_keep_wire = shift_keep_reg;
    shift_valid_wire = shift_valid_reg;
    shift_last_wire = shift_last_reg;
    shift_occupancy_wire = shift_occupancy_reg;
    
    // EXTRACT
    if(state == EXTRACT_STATE && (s_inbuf_axis_tready && s_inbuf_axis_tvalid)) begin
        shift_wire[(BUF_DATA_WIDTH  - 1) : 0] = (s_inbuf_axis_tdata >> EXTRACTED_STRUCT_WIDTH);
        shift_keep_wire [(BUF_KEEP_WIDTH  - 1) : 0] = (s_inbuf_axis_tkeep >> (EXTRACTED_STRUCT_WIDTH/8));
        
        // generate struct
        reg_extracted_axis_tdata = s_inbuf_axis_tdata[EXTRACTED_STRUCT_WIDTH-1 : 0];

        shift_occupancy_wire[0] = 1;
        // output stream, output struct size should not equal 0
        if(s_inbuf_axis_tlast) begin
            shift_valid_wire[0] = 1;
            shift_last_wire[0] = 1;
        end
    end
    else if(state == FORWARD_STATE && (s_inbuf_axis_tready && s_inbuf_axis_tvalid)) begin
        shift_wire[(BUF_DATA_WIDTH*2 - 1): (BUF_DATA_WIDTH*2 - EXTRACTED_STRUCT_WIDTH)] = s_inbuf_axis_tdata[EXTRACTED_STRUCT_WIDTH-1 : 0];
        shift_wire = shift_wire | (s_inbuf_axis_tdata >> EXTRACTED_STRUCT_WIDTH);

        shift_keep_wire[(BUF_KEEP_WIDTH*2 - 1): (BUF_KEEP_WIDTH*2 - EXTRACTED_STRUCT_WIDTH/8)] = s_inbuf_axis_tkeep[EXTRACTED_STRUCT_WIDTH/8-1 : 0];
        shift_keep_wire = shift_keep_wire | (s_inbuf_axis_tkeep >> (EXTRACTED_STRUCT_WIDTH/8));

        shift_occupancy_wire[0] = 1;
        if(!shift_valid_wire[1]) begin
            shift_valid_wire[1] = 1;
        end

        if(s_inbuf_axis_tlast) begin
            shift_valid_wire[0] = 1;
            shift_last_wire[0] = 1;
        end
    end
end

// output buf assignment
always @* begin
    reg_outbuf_axis_tdata = shift_wire[(BUF_DATA_WIDTH*2 - 1): BUF_DATA_WIDTH];
    reg_outbuf_axis_tkeep = shift_keep_wire[(BUF_KEEP_WIDTH*2 - 1): BUF_KEEP_WIDTH];
    reg_outbuf_axis_tlast = shift_last_wire[1];
    reg_outbuf_axis_tvalid = shift_valid_wire[1];
end


reg [EXTRACTED_STRUCT_WIDTH-1:0]    reg_extracted_axis_tdata;
reg                                 reg_extracted_axis_tvalid;
wire                                reg_extracted_axis_tready;

reg [BUF_DATA_WIDTH-1:0]           reg_outbuf_axis_tdata;
reg [BUF_KEEP_WIDTH-1:0]           reg_outbuf_axis_tkeep;
reg                                reg_outbuf_axis_tvalid;
wire                                reg_outbuf_axis_tready;
reg                                reg_outbuf_axis_tlast;

axis_register#(
    .DATA_WIDTH(BUF_DATA_WIDTH),
    .KEEP_ENABLE(1),
    .LAST_ENABLE(1),
    .USER_ENABLE(0),
    .REG_TYPE(2)
) reg1 (
    .clk(clk),
    .rst(rst),
    .s_axis_tdata(reg_outbuf_axis_tdata),
    .s_axis_tkeep(reg_outbuf_axis_tkeep),
    .s_axis_tlast(reg_outbuf_axis_tlast),
    .s_axis_tvalid(reg_outbuf_axis_tvalid),
    .s_axis_tready(reg_outbuf_axis_tready),
    .m_axis_tdata(m_outbuf_axis_tdata),
    .m_axis_tkeep(m_outbuf_axis_tkeep),
    .m_axis_tlast(m_outbuf_axis_tlast),
    .m_axis_tvalid(m_outbuf_axis_tvalid),
    .m_axis_tready(m_outbuf_axis_tready)
);

axis_register#(
    .DATA_WIDTH(EXTRACTED_STRUCT_WIDTH),
    .KEEP_ENABLE(0),
    .LAST_ENABLE(0),
    .USER_ENABLE(0),
    .REG_TYPE(2)
) reg2 (
    .clk(clk),
    .rst(rst),
    .s_axis_tdata(reg_extracted_axis_tdata),
    .s_axis_tvalid(reg_extracted_axis_tvalid),
    .s_axis_tready(reg_extracted_axis_tready),
    .m_axis_tdata(m_extracted_axis_tdata),
    .m_axis_tvalid(m_extracted_axis_tvalid),
    .m_axis_tready(m_extracted_axis_tready)
);

endmodule
