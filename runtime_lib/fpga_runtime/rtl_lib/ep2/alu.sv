module ALU #
(
    parameter LVAL_SIZE = 16,
    parameter RVAL_SIZE = 16,
    parameter RESULT_SIZE = 32,
    parameter OPID = 0 // 0 SUB, 1 ADD, 2 AND, 3, LT, 4 GT, 5 EQ, 6 LE, 7 GE
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [LVAL_SIZE-1:0]              s_lval_axis_tdata,
    input wire                              s_lval_axis_tvalid,
    output reg                              s_lval_axis_tready,

    input wire [RVAL_SIZE-1:0]              s_rval_axis_tdata,
    input wire                              s_rval_axis_tvalid,
    output reg                              s_rval_axis_tready,

    output reg [RESULT_SIZE-1:0]              m_val_axis_tdata,
    output reg                                m_val_axis_tvalid,
    input  wire                               m_val_axis_tready
);

always @* begin
    m_val_axis_tvalid = 0;
    s_lval_axis_tready = s_rval_axis_tvalid && m_val_axis_tready;
    s_rval_axis_tready = s_lval_axis_tvalid && m_val_axis_tready;
    m_val_axis_tvalid = s_lval_axis_tvalid && s_rval_axis_tvalid;
end

generate
    if (OPID == 0) begin
        always @* begin
            m_val_axis_tdata = s_lval_axis_tdata - s_rval_axis_tdata;
        end
    end 
    else if (OPID == 1) begin
        always @* begin
            m_val_axis_tdata = s_lval_axis_tdata + s_rval_axis_tdata;
        end
    end
    else if(OPID == 2) begin
        always @* begin
            m_val_axis_tdata = s_lval_axis_tdata && s_rval_axis_tdata;
        end
    end
    else if(OPID == 3) begin
        always @* begin
            m_val_axis_tdata = (s_lval_axis_tdata < s_rval_axis_tdata) ? 1 : 0;
        end
    end
    else if(OPID == 4) begin
        always @* begin
            m_val_axis_tdata = (s_lval_axis_tdata > s_rval_axis_tdata) ? 1 : 0;
        end
    end
    else if(OPID == 5) begin
        always @* begin
            m_val_axis_tdata = (s_lval_axis_tdata == s_rval_axis_tdata) ? 1 : 0;
        end
    end
    else if(OPID == 6) begin
        always @* begin
            m_val_axis_tdata = (s_lval_axis_tdata <= s_rval_axis_tdata) ? 1 : 0;
        end
    end
    else if(OPID == 7) begin
        always @* begin
            m_val_axis_tdata = (s_lval_axis_tdata >= s_rval_axis_tdata) ? 1 : 0;
        end
    end
endgenerate

endmodule