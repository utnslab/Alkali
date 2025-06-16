
`timescale 1ns / 1ps

module AES_engine #
(
    // Width of AXI data bus in bits
    parameter DATA_WIDTH = 512,
    parameter KEEP_WIDTH = (DATA_WIDTH/8)
)
(
    input  wire                       clk,
    input  wire                       rst,

    /*
    * Crossbar interface input
    */
    input  wire [DATA_WIDTH-1:0]            s_axis_tdata,
    input  wire [KEEP_WIDTH-1:0]            s_axis_tkeep,
    input  wire                             s_axis_tvalid,
    output wire                             s_axis_tready,
    input  wire                             s_axis_tlast,

    /*
    * Crossbar interface output
    */
    output  wire [DATA_WIDTH-1:0]             m_axis_tdata,
    output  wire [KEEP_WIDTH-1:0]             m_axis_tkeep,
    output  wire                              m_axis_tvalid,
    input   wire                              m_axis_tready,
    output  wire                              m_axis_tlast
    
);




reg  [3:0] f_512_to_128; // 0: no data 1: frist data
reg  [3:0] f_128_to_512;
reg [127:0] aes_in_data;
reg  [28:0] aes_state; // 29 cycle delay
wire [255:0] aes_key;
wire [127:0] aes_out;

assign aes_key = 256'h2b7e151628aed2a6abf7158809cf4f3c_762e7160f38b4da56a784d9045190cfe;

//////////////////aes descriptor////////////////////////

wire [128-1:0]                        m_aes_data_in_fifo_tdata;
wire [16-1:0]                         m_aes_data_in_fifo_tkeep;
wire                                  m_aes_data_in_fifo_tvalid;
wire                                  m_aes_data_in_fifo_tvalid_tmp;
wire                                  m_aes_data_in_fifo_tready;
wire                                  m_aes_data_in_fifo_tlast;

wire [128-1:0]                        s_aes_data_out_fifo_tdata;
wire                                  s_aes_data_out_fifo_tvalid;
wire                                  s_aes_data_out_fifo_tready;
wire [16-1:0]                         s_aes_data_out_fifo_tkeep;
wire                                  s_aes_data_out_fifo_tlast;

wire [DATA_WIDTH-1:0]         m_aes_data_out_fifo_tdata;
wire                                 m_aes_data_out_fifo_tvalid;
wire                                 m_aes_data_out_fifo_tready;
wire [KEEP_WIDTH-1:0]         m_aes_data_out_fifo_tkeep;
wire                                 m_aes_data_out_fifo_tlast;



///////////////////////////
wire [DATA_WIDTH-1:0]          s_aes_data_in_fifo_tdata;
wire [KEEP_WIDTH-1:0]          s_aes_data_in_fifo_tkeep;
wire                                  s_aes_data_in_fifo_tvalid;
wire                                  s_aes_data_in_fifo_tready;
wire                                  s_aes_data_in_fifo_tlast;

assign s_axis_tready = s_aes_data_in_fifo_tready;
assign s_aes_data_in_fifo_tdata  = s_axis_tdata;
assign s_aes_data_in_fifo_tvalid = s_axis_tvalid;
assign s_aes_data_in_fifo_tlast  = s_axis_tlast;
assign s_aes_data_in_fifo_tkeep  = s_axis_tkeep;


axis_fifo_adapter  #(
    .DEPTH(16 * KEEP_WIDTH),
    .S_DATA_WIDTH(DATA_WIDTH),
    .M_DATA_WIDTH(128),
    .S_KEEP_ENABLE (1),
    .M_KEEP_ENABLE (1),
    .S_KEEP_WIDTH (KEEP_WIDTH),
    .M_KEEP_WIDTH (16),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
aes_data_in_fifo (
    .clk(clk),
    .rst(rst),

    // AXI input
    .s_axis_tdata(s_aes_data_in_fifo_tdata),
    .s_axis_tvalid(s_aes_data_in_fifo_tvalid),
    .s_axis_tready(s_aes_data_in_fifo_tready),
    .s_axis_tlast(s_aes_data_in_fifo_tlast),
    .s_axis_tkeep(s_aes_data_in_fifo_tkeep),

    // AXI output
    .m_axis_tdata(m_aes_data_in_fifo_tdata),
    .m_axis_tvalid(m_aes_data_in_fifo_tvalid_tmp),
    .m_axis_tready(m_aes_data_in_fifo_tready),
    .m_axis_tlast(m_aes_data_in_fifo_tlast),
    .m_axis_tkeep(m_aes_data_in_fifo_tkeep)
);


wire                                 m_tkeep_tlast_fifo_tvalid;
wire                                 m_tkeep_tlast_fifo_tready;
wire [KEEP_WIDTH-1:0]         m_tkeep_tlast_fifo_tdata_tkeep;
wire                                 m_tkeep_tlast_fifo_tdata_tlast;
reg                                  m_tkeep_tlast_fifo_tready_reg;


assign m_tkeep_tlast_fifo_tready = m_tkeep_tlast_fifo_tready_reg;
axis_fifo #(
    .DEPTH(128),
    .DATA_WIDTH(KEEP_WIDTH+1),
    .KEEP_ENABLE(0),
    .LAST_ENABLE(0),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
tkeep_tlast_fifo (
    .clk(clk),
    .rst(rst),

    // AXI input
    .s_axis_tdata({m_aes_data_in_fifo_tkeep,m_aes_data_in_fifo_tlast}),
    .s_axis_tvalid(m_aes_data_in_fifo_tready && m_aes_data_in_fifo_tvalid),
    .s_axis_tready(m_aes_data_in_fifo_tready),

    // AXI output
    .m_axis_tdata({m_tkeep_tlast_fifo_tdata_tkeep,m_tkeep_tlast_fifo_tdata_tlast}),
    .m_axis_tvalid(m_tkeep_tlast_fifo_tvalid),
    .m_axis_tready(s_aes_data_out_fifo_tready && s_aes_data_out_fifo_tvalid)
);


aes_256 uut (
    .clk(clk), 
    .state(m_aes_data_in_fifo_tdata), 
    .key(aes_key), 
    .out(aes_out)
);


wire                                   s_aes_tiny_fifo_tready;
reg   [15:0]                           aes_tiny_fifo_counter;
wire                                   aes_tiny_fifo_half_full;

assign aes_tiny_fifo_half_full   = (aes_tiny_fifo_counter >= 32);
assign m_aes_data_in_fifo_tvalid = m_aes_data_in_fifo_tvalid_tmp && !aes_tiny_fifo_half_full;

always @(posedge clk) begin
    if(rst) begin
        aes_tiny_fifo_counter <= 0;
    end
    else begin
        if(aes_out && aes_state[0]) begin
            aes_tiny_fifo_counter = aes_tiny_fifo_counter + 1;
        end
        if(s_aes_data_out_fifo_tvalid && s_aes_data_out_fifo_tready) begin
            aes_tiny_fifo_counter = aes_tiny_fifo_counter - 1;
        end
    end
end

axis_fifo #(
    .DEPTH(64),
    .DATA_WIDTH(128),
    .KEEP_ENABLE(0),
    .LAST_ENABLE(0),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)
aes_tiny_fifo (
    .clk(clk),
    .rst(rst),

    // AXI input
    .s_axis_tdata(aes_out),
    .s_axis_tvalid(aes_state[0]),
    .s_axis_tready(s_aes_tiny_fifo_tready),

    // AXI output
    .m_axis_tdata(s_aes_data_out_fifo_tdata),
    .m_axis_tvalid(s_aes_data_out_fifo_tvalid),
    .m_axis_tready(s_aes_data_out_fifo_tready)
);

always @(posedge clk) begin
    if(rst) begin
        aes_state <= 0;
    end
    else begin
        aes_state = aes_state >> 1;
        aes_state[28] = m_aes_data_in_fifo_tvalid && m_aes_data_in_fifo_tready;
    end
end



axis_fifo_adapter  #(
    .DEPTH(16 * KEEP_WIDTH),
    .S_DATA_WIDTH(128),
    .M_DATA_WIDTH(DATA_WIDTH),
    .S_KEEP_ENABLE (1),
    .M_KEEP_ENABLE (1),
    .S_KEEP_WIDTH (16),
    .M_KEEP_WIDTH (KEEP_WIDTH),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .USER_ENABLE(0),
    .FRAME_FIFO(0)
)aes_data_out_fifo (
    .clk(clk),
    .rst(rst),

    // AXI input
    .s_axis_tdata(s_aes_data_out_fifo_tdata),
    .s_axis_tvalid(s_aes_data_out_fifo_tvalid),
    .s_axis_tready(s_aes_data_out_fifo_tready),
    .s_axis_tkeep(m_tkeep_tlast_fifo_tdata_tkeep),
    .s_axis_tlast(m_tkeep_tlast_fifo_tdata_tlast),

    // AXI output
    .m_axis_tdata(m_aes_data_out_fifo_tdata),
    .m_axis_tvalid(m_aes_data_out_fifo_tvalid),
    .m_axis_tready(m_aes_data_out_fifo_tready),
    .m_axis_tkeep(m_aes_data_out_fifo_tkeep),
    .m_axis_tlast(m_aes_data_out_fifo_tlast)
);


reg [7:0]  shape_counter;
axis_fifo #(
    .DEPTH(32 * KEEP_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),
    .KEEP_ENABLE(1),
    .KEEP_WIDTH(KEEP_WIDTH),
    .LAST_ENABLE(1),
    .USER_ENABLE(0),
    .ID_ENABLE(0),
    .DEST_ENABLE(0),
    .FRAME_FIFO(0)
)
aes_shape_fifo (
    .clk(clk),
    .rst(rst),

    // AXI input
    .s_axis_tdata(m_aes_data_out_fifo_tdata),
    .s_axis_tkeep(m_aes_data_out_fifo_tkeep),
    .s_axis_tvalid(m_aes_data_out_fifo_tvalid),
    .s_axis_tready(m_aes_data_out_fifo_tready),
    .s_axis_tlast(m_aes_data_out_fifo_tlast),

    // AXI output
    .m_axis_tdata(m_axis_tdata),
    .m_axis_tkeep(m_axis_tkeep),
    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready),
    .m_axis_tlast(m_axis_tlast)
);

always @ (posedge clk) begin
    if(rst) begin
        shape_counter <= 0;
    end
    else begin
        if(m_aes_data_out_fifo_tlast && m_aes_data_out_fifo_tready && m_aes_data_out_fifo_tvalid)
            shape_counter = shape_counter + 1;
        if(m_axis_tlast && m_axis_tready && m_axis_tvalid)
            shape_counter = shape_counter - 1;
    end
end


endmodule