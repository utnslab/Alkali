module cam_warpper #(
    parameter TABLE_SIZE = 16,
    parameter KEY_SIZE = 8,
    parameter VALUE_SIZE = 32,
    parameter UPDATE_USER_WIDTH = 4,
    parameter LOOKUP_USER_WIDTH = 4
)(

    input  wire                          clk,
    input  wire                          rst,
    // UPDATE = 1 MATCH + 1 SET
    input  wire  [KEY_SIZE -1:0]                 update_req_index,
    input  wire  [VALUE_SIZE-1:0]                update_req_data,
    input  wire                                  update_req_valid,
    output wire                                  update_req_ready,
    input  wire  [UPDATE_USER_WIDTH-1:0]         update_req_user,


    // LOOKUP = 1 MATCH 
    input  wire  [KEY_SIZE-1:0]                  lookup_req_index,
    input  wire                                  lookup_req_valid,
    output  wire                                 lookup_req_ready,
    input  wire  [LOOKUP_USER_WIDTH-1:0]         lookup_req_user,
    output  wire [VALUE_SIZE-1:0]                lookup_value_data,
    output  wire                                 lookup_value_valid,
    input wire                                   lookup_value_ready,
    input  wire  [LOOKUP_USER_WIDTH-1:0]         lookup_value_user
);

initial
$timeformat(-9, 2, " ns", 20);

always @(posedge clk) begin
    if(lookup_req_valid && lookup_req_ready) begin
        $display("lookup_req_index: %m, time is %0t", $time);
    end
    if(update_req_valid && update_req_ready) begin
        $display("update_req_index: %m, time is %0t", $time);
    end
end
localparam ARB_MAX_USER_WIDTH = UPDATE_USER_WIDTH > LOOKUP_USER_WIDTH? UPDATE_USER_WIDTH + 1 : LOOKUP_USER_WIDTH + 1;
localparam CAM_ADDR_WIDTH = $clog2(TABLE_SIZE);


wire  [(KEY_SIZE+VALUE_SIZE)-1:0]                  arb_in_lookup_dataindex;
wire  [(KEY_SIZE+VALUE_SIZE)-1:0]                  arb_in_update_dataindex;
wire  [ARB_MAX_USER_WIDTH-1:0]                         arb_in_lookup_user;
wire  [ARB_MAX_USER_WIDTH-1:0]                         arb_in_update_user;

assign arb_in_lookup_dataindex = {{VALUE_SIZE{1'b0}}, lookup_req_index};
assign arb_in_update_dataindex = {update_req_data, update_req_index};
assign arb_in_lookup_user = {{1'b0}, lookup_req_user};
assign arb_in_update_user = {{1'b1}, update_req_user};

wire  [KEY_SIZE+VALUE_SIZE-1:0]       match_req_dataindex;
wire                                  match_req_valid;
wire                                  match_req_ready;
wire  [ARB_MAX_USER_WIDTH-1:0]        match_req_user;


reg  [KEY_SIZE+VALUE_SIZE-1:0]       match_result_dataindex_reg;
reg                                  match_result_valid_reg;
reg                                  match_result_ready_reg;
reg  [ARB_MAX_USER_WIDTH-1:0]        match_result_user_reg;
wire  [CAM_ADDR_WIDTH-1:0]           match_result_addr_reg;
wire                                 match_result_if_match_reg;

// // TODO: better queue slection based on busy signal
// axis_arb_mux #(
//     .S_COUNT(2),
//     .DATA_WIDTH(KEY_SIZE+VALUE_SIZE),
//     .KEEP_ENABLE(0),
//     .USER_ENABLE(1),
//     .USER_WIDTH(ARB_MAX_USER_WIDTH),
//     .ARB_TYPE("PRIORITY")
// )
// match_mux (
//     .clk(clk),
//     .rst(rst),
//     // AXI input
//     .s_axis_tdata({arb_in_update_dataindex, arb_in_lookup_dataindex}),
//     .s_axis_tvalid({update_req_valid, lookup_req_valid}),
//     .s_axis_tready({update_req_ready, lookup_req_ready}),
//     .s_axis_tuser({arb_in_update_user, arb_in_lookup_user}),

//     // AXI output
//     .m_axis_tdata(match_req_dataindex),
//     .m_axis_tvalid(match_req_valid),
//     .m_axis_tready(match_req_ready),
//     .m_axis_tuser(match_req_user)
// );

axis_mux #(
    .S_COUNT(2),
    .DATA_WIDTH(KEY_SIZE+VALUE_SIZE),
    .KEEP_ENABLE(0),
    .USER_ENABLE(1),
    .USER_WIDTH(ARB_MAX_USER_WIDTH)
)
match_mux (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata({arb_in_update_dataindex, arb_in_lookup_dataindex}),
    .s_axis_tvalid({update_req_valid, lookup_req_valid}),
    .s_axis_tready({update_req_ready, lookup_req_ready}),
    .s_axis_tlast({1'b1, 1'b1}),
    .s_axis_tuser({arb_in_update_user, arb_in_lookup_user}),

    // AXI output
    .m_axis_tdata(match_req_dataindex),
    .m_axis_tvalid(match_req_valid),
    .m_axis_tready(match_req_ready),
    .m_axis_tuser(match_req_user),

    .enable(1'b1),
    .select(match_select)
);
reg match_select = 1;
reg pre_select;

always @* begin
    if(update_req_valid && !lookup_req_valid) begin
        match_select = 1;
    end 
    else if(!update_req_valid && lookup_req_valid)begin
        match_select = 0;
    end
    else begin
        if(stall_count > 0) begin
            match_select = 0;
        end
        else begin
            match_select = !pre_select;
        end
    end
end

always @(posedge clk) begin
    if(rst) begin
        pre_select <= 0;
    end
    else begin
        if(update_req_valid && update_req_ready) begin
            pre_select <= 1;
        end
        else if(lookup_req_valid && lookup_req_ready) begin
            pre_select <= 0;
        end
    end
end

reg [CAM_ADDR_WIDTH-1:0]    cam_write_addr;
reg [KEY_SIZE-1:0]          cam_write_key;
reg                         cam_write_enable;
wire                         cam_write_busy;
wire                         cam_write_ready;
wire [KEY_SIZE-1:0]          cam_compare_key;
wire [2**CAM_ADDR_WIDTH-1:0] cam_match_many;
wire [2**CAM_ADDR_WIDTH-1:0] cam_match_single;
wire [CAM_ADDR_WIDTH-1:0]    cam_match_addr;
wire                         cam_match;

assign cam_write_ready = !cam_write_busy ;
cam#(
.DATA_WIDTH(KEY_SIZE),
.ADDR_WIDTH(CAM_ADDR_WIDTH),
.CAM_STYLE("BRAM")
)cam(
	 .clk(clk), 
	 .rst(rst) ,
	//input buf
	.write_addr(cam_write_addr),
	.write_data(cam_write_key),
	.write_delete(0),
	.write_enable(cam_write_enable),
	.write_busy(cam_write_busy),
	//output buf
	.compare_data(cam_compare_key),
	.match_many(cam_match_many),
	.match_single(cam_match_single),
	.match_addr(cam_match_addr),
	.match(cam_match)
);

assign cam_compare_key = match_req_dataindex[0 +: KEY_SIZE];
assign match_req_ready = !stall && match_result_ready_reg ;

reg [7:0] stall_count;
// TODO: lookup stall based on address conflict with write
wire stall = stall_count > 0;
always @(posedge clk) begin
    if(rst) begin
        match_result_dataindex_reg = 0;
        match_result_valid_reg = 0;
        match_result_user_reg = 0;
        stall_count = 0;
    end
    else begin
        // add 16 cycle delay for input update req's match
        if(match_req_ready && match_req_valid && (match_req_user[ARB_MAX_USER_WIDTH-1]) == 1) begin
            match_result_dataindex_reg <= match_req_dataindex;
            match_result_valid_reg <= match_req_valid;
            match_result_user_reg <= match_req_user;
            // TODO: depends on what reqest
            stall_count <= 5;
        end
        else if(match_req_ready && match_req_valid && (match_req_user[ARB_MAX_USER_WIDTH-1]) == 0) begin
            match_result_dataindex_reg <= match_req_dataindex;
            match_result_valid_reg <= match_req_valid;
            match_result_user_reg <= match_req_user;
            // TODO: depends on what reqest
            stall_count <= 0;
        end
        else begin
            if(stall_count!= 0) begin
                stall_count <= stall_count -1;
            end
            match_result_dataindex_reg <= 0;
            match_result_valid_reg <= 0;
            match_result_user_reg <= 0;
        end
    end
end

// no delay for cam match output sig
assign match_result_if_match_reg = cam_match;
assign match_result_addr_reg = cam_match_addr;

reg [CAM_ADDR_WIDTH-1 : 0] free_slot_id;
always @(posedge clk) begin
    if(rst) begin
        free_slot_id = 0;
    end
    else begin
        if((!match_result_if_match_reg) && match_result_valid_reg && match_result_ready_reg && match_result_user_reg[ARB_MAX_USER_WIDTH-1]) begin
            free_slot_id <= free_slot_id + 1;
        end
    end

end

always @* begin
    match_result_ready_reg = match_result_user_reg[ARB_MAX_USER_WIDTH-1]? cam_write_ready : ram_read_req_ready;
    // Them cam implementation requires addr and key keep until busy finishes
    // cam_write_addr = 0;
    // cam_write_key = 0;
    cam_write_enable = 0;

    ram_read_req_addr = 0;
    ram_read_req_user = 0;
    ram_read_req_valid = 0;

    ram_write_req_data = 0;
    ram_write_req_addr = 0;
    ram_write_req_valid = 0;
    // matched results for update req
    if(match_result_ready_reg && match_result_valid_reg && match_result_user_reg[ARB_MAX_USER_WIDTH-1]) begin
        cam_write_addr = match_result_if_match_reg ? match_result_addr_reg : free_slot_id;
        cam_write_key = match_result_dataindex_reg[0 +: KEY_SIZE];
        cam_write_enable = 1;

        ram_write_req_data = match_result_dataindex_reg[KEY_SIZE +: VALUE_SIZE];
        ram_write_req_addr = cam_write_addr;
        ram_write_req_valid = 1;
    end
    // matched results for lookup req
    else if (match_result_ready_reg && match_result_valid_reg && !match_result_user_reg[ARB_MAX_USER_WIDTH-1]) begin
        ram_read_req_addr = match_result_if_match_reg ? match_result_addr_reg : 2; // TODO: Acceept Return NULL
        ram_read_req_user = match_result_user_reg;
        ram_read_req_valid = 1;
    end
end


reg  [CAM_ADDR_WIDTH-1:0]     ram_read_req_addr;
reg  [LOOKUP_USER_WIDTH-1:0]  ram_read_req_user;
reg                           ram_read_req_valid;
wire                          ram_read_req_ready;

reg  [VALUE_SIZE-1:0]         ram_write_req_data;
reg  [CAM_ADDR_WIDTH-1:0]     ram_write_req_addr;
reg                           ram_write_req_valid;
wire                           ram_write_req_ready; // should alwyas be 1;


ram #(
    .ADDR_WIDTH(CAM_ADDR_WIDTH),
    .DATA_WIDTH(VALUE_SIZE),
    .USER_WIDTH(LOOKUP_USER_WIDTH)
)
value_ram (
    .clk(clk),
    .rst(rst),
    // write interface
    .s_write_data(ram_write_req_data),
    .s_write_addr(ram_write_req_addr),
    .s_write_valid(ram_write_req_valid),
    .s_write_ready(ram_write_req_ready),

    // read interface
    .s_read_addr(ram_read_req_addr),
    .s_read_user(ram_read_req_user),
    .s_read_valid(ram_read_req_valid),
    .s_read_ready(ram_read_req_ready),
    .m_read_data(lookup_value_data),
    .m_read_valid(lookup_value_valid),
    .m_read_user(lookup_value_user),
    .m_read_ready(lookup_value_ready)
);


endmodule