module struct_access #
(
    parameter STRUCT_WIDTH = 16,
    parameter ACCESS_OFFSET = 0,
    parameter ACCESS_SIZE = 4
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [STRUCT_WIDTH-1:0]           s_struct_axis_tdata,
    input wire                              s_struct_axis_tvalid,
    output reg                              s_struct_axis_tready,

    // output wire [ACCESS_SIZE-1:0]            m_val_axis_tdata,
    // output wire                              m_val_axis_tvalid,
    // input  wire                              m_val_axis_tready,

    output reg [STRUCT_WIDTH-1:0]             m_val_axis_tdata,
    output reg                                m_val_axis_tvalid,
    input  wire                               m_val_axis_tready    
);


always @* begin
    m_val_axis_tdata =  s_struct_axis_tdata[ACCESS_OFFSET +: ACCESS_SIZE];
    m_val_axis_tvalid = s_struct_axis_tvalid;
    s_struct_axis_tready = m_val_axis_tready;
end

// always @* begin
//     reg_struct_axis_tvalid = s_struct_axis_tvalid && reg_val_axis_tready;
//     reg_val_axis_tvalid = s_struct_axis_tvalid && reg_struct_axis_tready;

//     s_struct_axis_tready = reg_val_axis_tready && reg_struct_axis_tready;

//     reg_struct_axis_tdata = s_struct_axis_tdata;
//     reg_val_axis_tdata = s_struct_axis_tdata[ACCESS_OFFSET +: ACCESS_SIZE];
// end


// reg [STRUCT_WIDTH-1:0]            reg_struct_axis_tdata;
// reg                               reg_struct_axis_tvalid;
// wire                               reg_struct_axis_tready;

// reg [ACCESS_SIZE-1:0]            reg_val_axis_tdata;
// reg                              reg_val_axis_tvalid;
// wire                              reg_val_axis_tready;

// axis_register#(
//     .DATA_WIDTH(STRUCT_WIDTH),
//     .KEEP_ENABLE(0),
//     .LAST_ENABLE(0),
//     .USER_ENABLE(0),
//     .REG_TYPE(2)
// )
// reg1(
//     .clk(clk),
//     .rst(rst),
//     .s_axis_tdata(reg_struct_axis_tdata),
//     .s_axis_tvalid(reg_struct_axis_tvalid),
//     .s_axis_tready(reg_struct_axis_tready),
//     .m_axis_tdata(m_struct_axis_tdata),
//     .m_axis_tvalid(m_struct_axis_tvalid),
//     .m_axis_tready(m_struct_axis_tready)
// );

// axis_register#(
//     .DATA_WIDTH(ACCESS_SIZE),
//     .KEEP_ENABLE(0),
//     .LAST_ENABLE(0),
//     .USER_ENABLE(0),
//     .REG_TYPE(2)
// ) reg2(
//     .clk(clk),
//     .rst(rst),
//     .s_axis_tdata(reg_val_axis_tdata),
//     .s_axis_tvalid(reg_val_axis_tvalid),
//     .s_axis_tready(reg_val_axis_tready),
//     .m_axis_tdata(m_val_axis_tdata),
//     .m_axis_tvalid(m_val_axis_tvalid),
//     .m_axis_tready(m_val_axis_tready)
// );

endmodule