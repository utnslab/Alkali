module ctrl_dispatcher#(
    parameter D_COUNT = 2,
    parameter DISPATCH_WIDTH =  $clog2(D_COUNT),
    parameter REPLICATED_OUT_NUM = 3
)
(
    input  wire                       clk,
    input  wire                       rst,
    

    output wire  [REPLICATED_OUT_NUM*DISPATCH_WIDTH-1:0]          m_dispatcher_tdata,
    output wire  [REPLICATED_OUT_NUM-1:0]                         m_dispatcher_tvalid,
    input  wire  [REPLICATED_OUT_NUM-1:0]                         m_dispatcher_tready
);

reg [9:0] rr_counter;
reg replicate_selector_tvalid;

always @(posedge clk) begin
    if(rst) begin
        rr_counter <= 0;
    end
    else begin
        replicate_selector_tvalid <= 1;
        if (replicate_selector_tvalid && replicate_selector_tready) begin
            if(rr_counter == (D_COUNT-1)) begin
                rr_counter <= 0;
            end
            else begin
                rr_counter <= rr_counter + 1;
            end
        end
    end
end

wire  [DISPATCH_WIDTH-1:0]         replicate_selector_tdata;
wire                             replicate_selector_tvalid;
wire                             replicate_selector_tready;

assign replicate_selector_tdata = rr_counter;

axis_replication#(
.DATA_WIDTH(DISPATCH_WIDTH),
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
	.m_axis_out_tdata(m_dispatcher_tdata),
	.m_axis_out_tvalid(m_dispatcher_tvalid),
	.m_axis_out_tready(m_dispatcher_tready)
);
endmodule