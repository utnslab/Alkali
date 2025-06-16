module struct_assign #
(
    parameter STRUCT_WIDTH = 16,
    parameter ASSIGN_OFFSET = 0,
    parameter ASSIGN_SIZE = 4,
    parameter IF_NO_REG = 0
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [STRUCT_WIDTH-1:0]           s_struct_axis_tdata,
    input wire                              s_struct_axis_tvalid,
    output reg                              s_struct_axis_tready,

    input  wire [ASSIGN_SIZE-1:0]           s_assignv_axis_tdata,
    input wire                              s_assignv_axis_tvalid,
    output reg                              s_assignv_axis_tready,

    output wire [STRUCT_WIDTH-1:0]             m_struct_axis_tdata,
    output wire                                m_struct_axis_tvalid,
    input  wire                                m_struct_axis_tready    
);
reg  [STRUCT_WIDTH-1:0] temp_struct;

always @* begin
    reg_struct_axis_tdata = 0;
    reg_struct_axis_tvalid = s_struct_axis_tvalid && s_assignv_axis_tvalid;
    s_struct_axis_tready = reg_struct_axis_tready && s_assignv_axis_tvalid;
    s_assignv_axis_tready = reg_struct_axis_tready && s_struct_axis_tvalid;
    
    if(s_struct_axis_tvalid && s_struct_axis_tready && s_assignv_axis_tvalid && s_assignv_axis_tready) begin
        temp_struct = s_struct_axis_tdata;
        temp_struct[ASSIGN_OFFSET +: ASSIGN_SIZE] = s_assignv_axis_tdata;
        reg_struct_axis_tdata = temp_struct;
    end
end


reg [STRUCT_WIDTH-1:0]             reg_struct_axis_tdata;
reg                                reg_struct_axis_tvalid;
wire                                reg_struct_axis_tready;   

generate 
    if(IF_NO_REG == 1) begin
        assign m_struct_axis_tdata = reg_struct_axis_tdata;
        assign m_struct_axis_tvalid = reg_struct_axis_tvalid;
        assign reg_struct_axis_tready = m_struct_axis_tready;
    end else begin
        axis_register#(
            .DATA_WIDTH(STRUCT_WIDTH),
            .KEEP_ENABLE(0),
            .LAST_ENABLE(0),
            .USER_ENABLE(0),
            .REG_TYPE(2)
        ) reg1 (
            .clk(clk),
            .rst(rst),
            .s_axis_tdata(reg_struct_axis_tdata),
            .s_axis_tvalid(reg_struct_axis_tvalid),
            .s_axis_tready(reg_struct_axis_tready),
            .m_axis_tdata(m_struct_axis_tdata),
            .m_axis_tvalid(m_struct_axis_tvalid),
            .m_axis_tready(m_struct_axis_tready)
        );
    end
endgenerate


endmodule