module ram #(
	parameter ADDR_WIDTH = 8,
	parameter DATA_WIDTH = 8,
    parameter USER_WIDTH = 8,
	parameter RAM_STYLE = "block"
)
(
	input wire clk,
	input wire rst,
	input wire [DATA_WIDTH-1:0] s_write_data,
	input wire [ADDR_WIDTH-1:0] s_write_addr,
    input wire  s_write_valid,
    output wire s_write_ready,

	input wire [ADDR_WIDTH-1:0] s_read_addr,
    input wire [USER_WIDTH-1:0] s_read_user,
	input wire s_read_valid,
    output reg s_read_ready,
	output reg [DATA_WIDTH-1:0] m_read_data,
    output reg                  m_read_valid,
    output reg [USER_WIDTH-1:0] m_read_user,
    input wire m_read_ready
    
);

(* ram_style = RAM_STYLE *)
reg [DATA_WIDTH-1:0]ram[2**ADDR_WIDTH-1:0];
integer i;

initial begin
	for (i = 0; i < 2**ADDR_WIDTH; i = i + 1) begin
		ram[i] = 0;
	end
end

assign s_write_ready = 1;
always @(posedge clk) begin
	if (s_write_valid && s_write_ready) begin
		ram[s_write_addr] <= s_write_data;
	end

    s_read_ready <= m_read_ready;

    if(s_read_ready && s_read_valid) begin
        m_read_data <= ram[s_read_addr];
        m_read_user <= s_read_user;
        m_read_valid <= s_read_valid && s_read_ready;
    end
	else begin
        m_read_data <= 0;
        m_read_user <= 0;
        m_read_valid <= 0;
	end
end

endmodule