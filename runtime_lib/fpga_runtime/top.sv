`timescale 1ns / 1ps

module packet_gen_parallel;

reg  clk;
reg rst;
reg start;
reg [9:0] pcount;
reg [9:0] ocount;
always begin
    clk = ~clk; 
    #2;
end

initial begin
    clk = 0;
    rst = 1;
    packet_len = 4;
    start = 0;
    
    #1000;
    rst = 0;
    $display("--------------\nRST, Start Sending %4dB packet", packet_len * 64 );
    #500;
    start = 1;
end

localparam AXIS_DATA_WIDTH = 512;
localparam AXIS_KEEP_WIDTH = AXIS_DATA_WIDTH / 8;
localparam PORTS = 1;
localparam CTX_DATA_WIDTH = 16;

reg  [PORTS*AXIS_DATA_WIDTH-1:0]    packet_axis_tdata;
reg  [PORTS*AXIS_KEEP_WIDTH-1:0]    packet_axis_tkeep;
reg  [PORTS-1:0]                    packet_axis_tvalid;
wire [PORTS-1:0]                    packet_axis_tready;
reg  [PORTS-1:0]                    packet_axis_tlast;

wire out_valid;
ep2top #
(   

)
ep2top
(
    .clk(clk),
    .rst(rst),

    /*
    * Receive data from the wire
    */
    .NET_RECV_0_tdata (packet_axis_tdata),
    .NET_RECV_0_tvalid(packet_axis_tvalid),
    .NET_RECV_0_tlast (packet_axis_tlast),
    .NET_RECV_0_tkeep (packet_axis_tkeep),
    .NET_RECV_0_tready(packet_axis_tready),

    .NET_SEND_0_tready(1),
    .NET_SEND_0_tvalid(out_valid),
    .NET_SEND_1_tready(1),
    .NET_SEND_2_tready(1),
    .NET_SEND_3_tready(1),
    .NET_SEND_4_tready(1),
    
    .DMA_WRITE_0_tready(1),
    .DMA_WRITE_1_tready(1),
    .DMA_WRITE_2_tready(1),
    .DMA_WRITE_3_tready(1),
    .DMA_WRITE_4_tready(1)
    
);



reg [63:0] c_counter;

reg [15:0] packet_len;
wire [15:0] header_length;

assign header_length = (packet_len)*64 - 14;

always@(*) begin

    packet_axis_tvalid = 0;
    packet_axis_tlast = 0;
    if(!rst && pcount < 64 && start) begin
        packet_axis_tvalid = 1;
        if(c_counter == 0) begin
            packet_axis_tdata = 512'h1514131211100F0E0D0C0B0A0908070605040302010081B90801020001006501A8C06401A8C0B7B5114000400000F2050045000855545352515AD5D4D3D2D1DA; // udp header
            packet_axis_tdata[16*8 +: 8] = header_length[15:8];
            packet_axis_tdata[17*8 +: 8] = header_length[7:0];
            packet_axis_tkeep = {64{1'b1}};
        end
        else begin
            packet_axis_tdata = c_counter;  
            packet_axis_tkeep = {64{1'b1}};
        end
        if(c_counter == packet_len-1) begin
            packet_axis_tkeep = {64{1'b1}};
            packet_axis_tlast = 1;
        end
    end
end

always@(posedge clk) begin
    if(rst) begin
        c_counter <= 0;
        pcount <= 0 ;
        ocount <= 0;
    end
    else begin
        if(packet_axis_tready && packet_axis_tvalid) begin
            c_counter<= c_counter+1;
            if(c_counter == packet_len-1) begin
                c_counter <= 0;
                pcount <= pcount + 1;
            end
        end
        if(out_valid) begin
            ocount <= ocount + 1;
        end

    end
end

endmodule
