// module reg_validdata #
// (
//     parameter DATA_WIDTH = 16,
//     parameter IF_KEEP = 0,
//     parameter IF_LAST = 0,
// )
// (
//     input  wire                       clk,
//     input  wire                       rst,
    
//     input wire [DATA_WIDTH-1:0]           s_axis_tdata,
//     input wire                            s_axis_tvalid,
//     output reg                             s_axis_tready,

//     output reg [DATA_WIDTH-1:0]            m_axis_tdata,
//     output reg                             m_axis_tvalid,
//     input wire                              m_axis_tready  
// );


// // valid path
// always @(posedge clk) begin
//     // if (m_axis_tvalid && m_axis_tready) begin
//     if (m_axis_tvalid && s_axis_tready) begin
//         m_axis_tdata <= s_axis_tdata;
//         m_axis_tvalid <= s_axis_tvalid;
//     end
// end


// // ready path
// always @* begin
//     s_axis_tready = m_axis_tready;
// end



// endmodule