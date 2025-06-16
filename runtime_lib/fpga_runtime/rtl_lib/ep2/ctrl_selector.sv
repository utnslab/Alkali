module ctrl_selector#(
    parameter S_COUNT = 2,
    parameter SELECT_WIDTH =  $clog2(S_COUNT),
    parameter REPLICATED_OUT_NUM = 3
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [S_COUNT-1:0]         s_inc_tdata,
    input wire [S_COUNT-1:0]         s_inc_tvalid,
    output reg [S_COUNT-1:0]         s_inc_tready,

    output wire  [REPLICATED_OUT_NUM*SELECT_WIDTH-1:0]            m_selector_tdata,
    output wire  [REPLICATED_OUT_NUM-1:0]                         m_selector_tvalid,
    input  wire  [REPLICATED_OUT_NUM-1:0]                         m_selector_tready
);

// at most 256 inflights
reg [7:0] selector_valid_count_array [S_COUNT-1:0];

integer j;

initial begin
    for (j = 0; j < S_COUNT; j = j + 1) begin
        selector_valid_count_array[j] = 0;
    end
end

genvar  rid;
generate
    for (rid=0; rid<S_COUNT; rid = rid + 1) begin: inport

        wire inc;
        wire dec;
        assign inc = s_inc_tdata[rid];
        assign dec = replicate_selector_tvalid && replicate_selector_tready && (replicate_selector_tdata == rid);

        always @(posedge clk) begin
            if(inc && !dec) begin
                selector_valid_count_array[rid] = selector_valid_count_array[rid] + 1;
            end
            else if (!inc && dec) begin
                selector_valid_count_array[rid] = selector_valid_count_array[rid] - 1;
            end
        end

        assign arbiter_valid_array[rid] = (selector_valid_count_array[rid] > 0) & replicate_selector_tready;
    end
endgenerate

wire [S_COUNT-1:0]         arbiter_valid_array;

// arbiter instance
arbiter #(
    .PORTS(S_COUNT),
    .TYPE("ROUND_ROBIN"),
    .BLOCK("NONE"),
    .LSB_PRIORITY("HIGH")
)
arb_inst (
    .clk(clk),
    .rst(rst),

    .request(arbiter_valid_array & ~ grant),
    .acknowledge(1),
    .grant(grant),
    .grant_valid(replicate_selector_tvalid),
    .grant_encoded(replicate_selector_tdata)
);

wire [S_COUNT-1:0]         grant;
wire  [SELECT_WIDTH-1:0]         replicate_selector_tdata;
wire                             replicate_selector_tvalid;
wire                             replicate_selector_tready;


axis_replication#(
.DATA_WIDTH(SELECT_WIDTH),
.IF_STREAM(0),
.REAPLICA_COUNT(REPLICATED_OUT_NUM)
)axis_replication_6(
	 .clk(clk), 
	 .rst(rst) ,
	//
	.s_axis_in_tdata(replicate_selector_tdata),
	.s_axis_in_tvalid(replicate_selector_tvalid),
	.s_axis_in_tready(replicate_selector_tready),
	//
	.m_axis_out_tdata(m_selector_tdata),
	.m_axis_out_tvalid(m_selector_tvalid),
	.m_axis_out_tready(m_selector_tready)
);
endmodule