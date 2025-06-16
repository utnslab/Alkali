module ctrl_barrier#(
    parameter PORT_COUNT = 2
)
(
    input  wire                       clk,
    input  wire                       rst,
    
    input wire [PORT_COUNT-1:0]         s_inc,

    input wire [PORT_COUNT-1:0]         s_dec,

    output wire  [PORT_COUNT:0]         ctrl_barrier
);


integer j, c;

reg [15:0] incount_array [PORT_COUNT-1:0];
reg [15:0] outcount_array [PORT_COUNT-1:0];
initial begin
    for (j = 0; j < PORT_COUNT; j = j + 1) begin
        incount_array[j] = 0;
        outcount_array[j] = 0;
    end
end

reg[15:0] min_incount;
always @(*) begin
  min_incount = incount_array[0];
  for (c = 0; c <= PORT_COUNT; c++)
  begin
    if (incount_array[c] < min_incount)
    begin
       min_incount  = incount_array[c];
    end
  end
end

genvar  rid;
generate
    for (rid=0; rid<PORT_COUNT; rid = rid + 1) begin: inport

        wire inc;
        wire dec;
        assign inc = s_inc[rid];
        assign dec = s_dec[rid];

        always @(posedge clk) begin
            if(inc) begin
                incount_array[rid] <= incount_array[rid] + 1;
            end
            else if (dec) begin
                outcount_array[rid] = outcount_array[rid] - 1;
            end
        end

        assign ctrl_barrier[rid] = (outcount_array[rid] < min_incount);
    end
endgenerate

endmodule