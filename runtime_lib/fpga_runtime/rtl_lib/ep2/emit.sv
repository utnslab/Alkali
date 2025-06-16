module emit #
(
    // Width of AXI stream interfaces in bits
    parameter BUF_DATA_WIDTH = 256,
    // AXI stream tkeep signal width (words per cycle)
    parameter BUF_KEEP_WIDTH = (BUF_DATA_WIDTH/8),
    parameter IF_INPUT_BUF = 1,
    parameter INPUT_BUF_STRUCT_WIDTH = 256,
    parameter INPUT_BUF_STRUCT_KEEP_WIDTH = (INPUT_BUF_STRUCT_WIDTH/8)
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
    input wire [INPUT_BUF_STRUCT_WIDTH-1:0]         s_struct_axis_tdata,
    input wire [INPUT_BUF_STRUCT_KEEP_WIDTH-1:0]    s_struct_axis_tkeep,
    input wire                                      s_struct_axis_tvalid,
    output  reg                                     s_struct_axis_tready,
    input wire                                      s_struct_axis_tlast,

    output reg [BUF_DATA_WIDTH-1:0]           m_outbuf_axis_tdata,
    output reg [BUF_KEEP_WIDTH-1:0]           m_outbuf_axis_tkeep,
    output reg                                m_outbuf_axis_tvalid,
    input  wire                               m_outbuf_axis_tready,
    output reg                                m_outbuf_axis_tlast
);

wire [INPUT_BUF_STRUCT_KEEP_WIDTH-1:0]    wrapped_s_struct_axis_tkeep;
wire                                      wrapped_s_struct_axis_tlast;

assign wrapped_s_struct_axis_tkeep = IF_INPUT_BUF ? s_struct_axis_tkeep : ((1<<BUF_KEEP_WIDTH) - 1);
assign wrapped_s_struct_axis_tlast = IF_INPUT_BUF ? s_struct_axis_tlast : 1;

localparam EMIT_BUF_STATE = 0;
localparam EMIT_STRUCT_STATE = 1;
reg [1:0] state;

reg [1:0] shift_occupancy_reg;
reg [1:0] shift_valid_reg;
reg [1:0] shift_last_reg;
reg [(BUF_DATA_WIDTH *2 - 1):0] shift_reg;
reg [(BUF_KEEP_WIDTH *2 - 1):0] shift_keep_reg;

reg [1:0] shift_occupancy_wire;
reg [1:0] shift_valid_wire;
reg [1:0] shift_last_wire;
wire [9:0] count_one_reg;
reg [(BUF_DATA_WIDTH *2 - 1):0] shift_wire;
reg [(BUF_KEEP_WIDTH *2 - 1):0] shift_keep_wire;

reg [(BUF_KEEP_WIDTH - 1):0] struct_shift_count;

reg [(BUF_DATA_WIDTH - 1):0] shift_tmp;
reg [(BUF_KEEP_WIDTH - 1):0] shift_keep_tmp;

reg [(BUF_DATA_WIDTH - 1):0] shift_debug_1;
reg [(BUF_DATA_WIDTH - 1):0] shift_debug_0;
assign shift_debug_1 = shift_wire[(BUF_DATA_WIDTH *2 - 1) : BUF_DATA_WIDTH];
assign shift_debug_0 = shift_wire[(BUF_DATA_WIDTH  - 1) : 0];


assign count_one_reg = $countones(s_inbuf_axis_tkeep);
always @(posedge clk) begin
    if(rst) begin
        shift_reg = 0;
        shift_keep_reg = 0;
        shift_valid_reg = 0;
        shift_last_reg = 0;
        shift_occupancy_reg = 0;
    end
    else begin
        // TODO: Check whether this use wire to driven is correct or not
        if(!shift_occupancy_wire[1] || (m_outbuf_axis_tready && m_outbuf_axis_tvalid)) begin
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
        state <= EMIT_BUF_STATE;
    end
    else begin

        if(state == EMIT_BUF_STATE && (s_inbuf_axis_tready && s_inbuf_axis_tvalid)) begin
            if(s_inbuf_axis_tlast) begin
                state <= EMIT_STRUCT_STATE;
            end
        end
        else if(state == EMIT_STRUCT_STATE && (s_struct_axis_tvalid && s_struct_axis_tready && wrapped_s_struct_axis_tlast)) begin
            state <= EMIT_BUF_STATE;
        end
    end
end

always @* begin
    s_inbuf_axis_tready = 0;
    s_struct_axis_tready = 0;
    if(state == EMIT_BUF_STATE) begin
        s_inbuf_axis_tready = !shift_occupancy_reg[0];
    end
    else begin
        s_struct_axis_tready = !shift_occupancy_reg[0] && m_outbuf_axis_tready;
    end

    shift_wire = shift_reg;
    shift_keep_wire = shift_keep_reg;
    shift_valid_wire = shift_valid_reg;
    shift_last_wire = shift_last_reg;
    shift_occupancy_wire = shift_occupancy_reg;
    shift_tmp = 0;
    
    // EMIT BUF
    if(state == EMIT_BUF_STATE && (s_inbuf_axis_tready && s_inbuf_axis_tvalid)) begin
        shift_wire[(BUF_DATA_WIDTH  - 1) : 0] =  s_inbuf_axis_tdata;
        shift_keep_wire[(BUF_KEEP_WIDTH  - 1) : 0] = s_inbuf_axis_tkeep;

        if(s_inbuf_axis_tkeep != 0)
            shift_occupancy_wire[0] = 1;
        // If stream is full, emit is valid
        if(s_inbuf_axis_tkeep == ((1 << BUF_KEEP_WIDTH) - 1)) begin
            shift_valid_wire[0] = 1;
            shift_last_wire[0] = 1;
            struct_shift_count = 0;
        end
        else begin
            // how many bits to shift form the struct flow
            struct_shift_count = 8 * (BUF_KEEP_WIDTH - count_one_reg);
        end
    end
    else if(state == EMIT_STRUCT_STATE && (s_struct_axis_tready && s_struct_axis_tvalid)) begin
        // put lo struct_shift_count bits into hi
        shift_tmp = s_struct_axis_tdata << (BUF_DATA_WIDTH - struct_shift_count); 
        // put hi struct_shift_count bits into shift wire left part hi
        shift_wire[(BUF_DATA_WIDTH*2 - 1): BUF_DATA_WIDTH] = shift_wire[(BUF_DATA_WIDTH*2 - 1): BUF_DATA_WIDTH] | shift_tmp;
        shift_wire = shift_wire | (s_struct_axis_tdata >> struct_shift_count);


        // put lo struct_shift_count bits into hi
        shift_keep_tmp = wrapped_s_struct_axis_tkeep << (BUF_KEEP_WIDTH - struct_shift_count/8); 
        shift_keep_wire[(BUF_KEEP_WIDTH*2 - 1): BUF_KEEP_WIDTH] = shift_keep_wire[(BUF_KEEP_WIDTH*2 - 1): BUF_KEEP_WIDTH] | shift_keep_tmp;
        shift_keep_wire = shift_keep_wire | (wrapped_s_struct_axis_tkeep >> (struct_shift_count/8));

        if((wrapped_s_struct_axis_tkeep >> (struct_shift_count/8)) != 0) begin
            shift_occupancy_wire[0] = 1;
            if(wrapped_s_struct_axis_tlast) begin
                shift_valid_wire[0] = 1;
                shift_last_wire[0] = 1;
            end
        end

        if(shift_keep_tmp != 0) begin
            shift_valid_wire[1] = 1;
            shift_occupancy_wire[1] = 1;
            if(wrapped_s_struct_axis_tlast && shift_last_wire[0] != 1) begin
                shift_last_wire[1] = 1;
            end
        end
    end
end


// output buf assignment
always @* begin
    m_outbuf_axis_tdata = shift_wire[(BUF_DATA_WIDTH*2 - 1): BUF_DATA_WIDTH];
    m_outbuf_axis_tkeep = shift_keep_wire[(BUF_KEEP_WIDTH*2 - 1): BUF_KEEP_WIDTH];
    m_outbuf_axis_tlast = shift_last_wire[1];
    m_outbuf_axis_tvalid = shift_valid_wire[1];
end

// reg emit_state [1:0] state; 
// localparam EMIT_BUF = 1;
// localparam EMIT_STURCT = 1;

// always @(posedge clk) begin
//     if(rst) begin
//         state <= 0;
//     end
//     else begin
//         if(state ==0 && s_inbuf_axis_tready && s_inbuf_axis_tvalid && s_inbuf_axis_tlast) begin
//             state <= 1;
//         end
//         else if(state = 1 && s_emit_axis_tvalid && s_emit_axis_tready && s_emit_axis_tlast) begin
//             state <= 0;
//         end


//     end

// end
// always @* begin
//     s_inbuf_axis_tready = m_outbuf_axis_tready && (state == 0);
//     s_emit_axis_tready = m_outbuf_axis_tready && (state == 1);

//     m_outbuf_axis_tvalid = 0;
//     m_outbuf_axis_tdata = 0;
//     m_outbuf_axis_tkeep = 0;
//     m_outbuf_axis_tlast = 0;
    
//     if(state ==0 && s_inbuf_axis_tready && s_inbuf_axis_tvalid && s_inbuf_axis_tlast) begin
//         // TODO: change this into a stage machine to support multi cycle input stream
//         m_extracted_axis_tdata = s_inbuf_axis_tdata[0 +: EXTRACTED_STRUCT_WIDTH];
//         m_extracted_axis_tvalid = s_inbuf_axis_tvalid;

//         // shift
//         m_outbuf_axis_tdata =  s_inbuf_axis_tdata << EXTRACTED_STRUCT_WIDTH;
//         m_outbuf_axis_tkeep = s_inbuf_axis_tkeep << (EXTRACTED_STRUCT_WIDTH/8);
//         m_outbuf_axis_tvalid = s_inbuf_axis_tvalid; 
//         m_outbuf_axis_tlast = s_inbuf_axis_tlast;
//     end
// end

endmodule
