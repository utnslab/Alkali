module cam_arbiter #
(
    parameter TABLE_SIZE = 16,
    parameter KEY_SIZE = 8,
    parameter VALUE_SIZE = 32,
    // Number of lookup port inputs
    parameter LOOKUP_PORTS = 2,
    // Number of update port inputs
    parameter UPDATE_PORTS = 2
)(

    input  wire                          clk,
    input  wire                          rst,

    /*
     * LOOKUP source inputs
     */
    input  wire  [LOOKUP_PORTS*KEY_SIZE-1:0]   s_lookup_req_index,
    input  wire  [LOOKUP_PORTS-1:0]            s_lookup_req_valid,
    output  wire  [LOOKUP_PORTS-1:0]            s_lookup_req_ready,

    output  reg [LOOKUP_PORTS*VALUE_SIZE-1:0]  s_lookup_value_data,
    output  reg [LOOKUP_PORTS-1:0]             s_lookup_value_valid,
    input wire   [LOOKUP_PORTS-1:0]             s_lookup_value_ready,

    /*
     * UPDATE source inputs
     */
    input   wire  [UPDATE_PORTS*KEY_SIZE-1:0]   s_update_req_index,
    input   wire  [UPDATE_PORTS-1:0]            s_update_req_index_valid,
    output  wire [UPDATE_PORTS-1:0]             s_update_req_index_ready,
    input   wire  [UPDATE_PORTS*VALUE_SIZE-1:0] s_update_req_data,
    input   wire  [UPDATE_PORTS-1:0]            s_update_req_data_valid,
    output  wire  [UPDATE_PORTS-1:0]            s_update_req_data_ready

);
localparam  UPDATE_USER_WIDTH = 4;
localparam  LOOKUP_USER_WIDTH = 4;
reg  [UPDATE_PORTS*(KEY_SIZE + VALUE_SIZE)-1:0]   merged_update_req_indexdata;
reg  [UPDATE_PORTS*(KEY_SIZE + VALUE_SIZE)-1:0]   merged_update_req_indexdata;
reg  [UPDATE_PORTS-1:0]            merged_update_req_valid;
reg  [UPDATE_PORTS-1:0]            merged_update_req_ready;
reg  [UPDATE_PORTS*UPDATE_USER_WIDTH-1:0]            merged_update_req_user;

genvar  uportid;
generate
    for (uportid=0; uportid<UPDATE_PORTS; uportid = uportid + 1) begin: uport_merge
        wire  [KEY_SIZE-1:0]    fifo_update_req_index;
        wire                    fifo_update_req_index_valid;
        reg                     fifo_update_req_index_ready;
        wire  [VALUE_SIZE-1:0]  fifo_update_req_data;
        wire                    fifo_update_req_data_valid;
        reg                     fifo_update_req_data_ready;

        axis_fifo #(
            .DEPTH(4),
            .DATA_WIDTH(KEY_SIZE),
            .KEEP_ENABLE(0),
            .LAST_ENABLE(0),
            .ID_ENABLE(0),
            .DEST_ENABLE(0),
            .USER_ENABLE(0),
            .FRAME_FIFO(0)
        )
        update_fifo_index (
            .clk(clk),
            .rst(rst),
            // AXI input
            .s_axis_tdata(s_update_req_index[uportid*KEY_SIZE +: KEY_SIZE]),
            .s_axis_tvalid(s_update_req_index_valid[uportid]),
            .s_axis_tready(s_update_req_index_ready[uportid]),

            // AXI output
            .m_axis_tdata(fifo_update_req_index),
            .m_axis_tvalid(fifo_update_req_index_valid),
            .m_axis_tready(fifo_update_req_index_ready)
        );


        axis_fifo #(
            .DEPTH(4),
            .DATA_WIDTH(VALUE_SIZE),
            .KEEP_ENABLE(0),
            .LAST_ENABLE(0),
            .ID_ENABLE(0),
            .DEST_ENABLE(0),
            .USER_ENABLE(0),
            .FRAME_FIFO(0)
        )
        update_fifo_data (
            .clk(clk),
            .rst(rst),
            // AXI input
            .s_axis_tdata(s_update_req_data[uportid*VALUE_SIZE +: VALUE_SIZE]),
            .s_axis_tvalid(s_update_req_data_valid[uportid]),
            .s_axis_tready(s_update_req_data_ready[uportid]),

            // AXI output
            .m_axis_tdata(fifo_update_req_data),
            .m_axis_tvalid(fifo_update_req_data_valid),
            .m_axis_tready(fifo_update_req_data_ready)
        );

        always @* begin
            merged_update_req_user[uportid*(UPDATE_USER_WIDTH) +: UPDATE_USER_WIDTH] = uportid;
            merged_update_req_indexdata[uportid*(KEY_SIZE+VALUE_SIZE) +: (KEY_SIZE+VALUE_SIZE)] = {fifo_update_req_index, fifo_update_req_data};
            merged_update_req_valid[uportid] = fifo_update_req_index_valid && fifo_update_req_data_valid;
            fifo_update_req_index_ready = merged_update_req_ready[uportid] && fifo_update_req_index_valid && fifo_update_req_data_valid;
            fifo_update_req_data_ready = merged_update_req_ready[uportid] && fifo_update_req_index_valid && fifo_update_req_data_valid;
        end
    end
endgenerate


wire  [KEY_SIZE + VALUE_SIZE-1:0]     selected_update_req_indexdata;
wire                                  selected_update_req_valid;
wire                                  selected_update_req_ready;
wire  [UPDATE_USER_WIDTH-1:0]         selected_update_req_user;

axis_arb_mux #(
    .S_COUNT(UPDATE_PORTS),
    .DATA_WIDTH(KEY_SIZE+VALUE_SIZE),
    .KEEP_ENABLE(0),
    .USER_ENABLE(1),
    .USER_WIDTH(UPDATE_USER_WIDTH),
    .ARB_TYPE("ROUND_ROBIN")
)
update_mux (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(merged_update_req_indexdata),
    .s_axis_tvalid(merged_update_req_valid),
    .s_axis_tready(merged_update_req_ready),
    .s_axis_tuser(merged_update_req_user),

    // AXI output
    .m_axis_tdata(selected_update_req_indexdata),
    .m_axis_tvalid(selected_update_req_valid),
    .m_axis_tready(selected_update_req_ready),
    .m_axis_tuser(selected_update_req_user)
);

wire  [KEY_SIZE-1:0]                  selected_lookup_req_index;
wire                                  selected_lookup_req_valid;
wire                                  selected_lookup_req_ready;
wire  [LOOKUP_USER_WIDTH-1:0]         selected_lookup_req_user;
reg  [LOOKUP_PORTS*LOOKUP_USER_WIDTH-1:0]      lookup_req_user;

wire  [VALUE_SIZE-1:0]                lookup_result_data;
wire                                  lookup_result_valid;
reg                                   lookup_result_ready;
wire  [LOOKUP_USER_WIDTH-1:0]         lookup_result_user;

wire  [LOOKUP_PORTS*KEY_SIZE-1:0]   fifo_lookup_req_index;
wire  [LOOKUP_PORTS-1:0]            fifo_lookup_req_valid;
wire  [LOOKUP_PORTS-1:0]            fifo_lookup_req_ready;
genvar  lportid;
generate
    for (lportid=0; lportid<LOOKUP_PORTS; lportid = lportid + 1) begin: lookup_user_gen
        always @(posedge clk) begin
            lookup_req_user[lportid*(LOOKUP_USER_WIDTH) +: LOOKUP_USER_WIDTH] <= lportid;
        end
        axis_register #(
            .DATA_WIDTH(KEY_SIZE),
            .KEEP_ENABLE(0),
            .LAST_ENABLE(0),
            .ID_ENABLE(0),
            .DEST_ENABLE(0),
            .USER_ENABLE(0),
            .REG_TYPE(2)
        )
        lookupreq_reg (
            .clk(clk),
            .rst(rst),
            // AXI input
            .s_axis_tdata(s_lookup_req_index[lportid*KEY_SIZE +: KEY_SIZE]),
            .s_axis_tvalid(s_lookup_req_valid[lportid]),
            .s_axis_tready(s_lookup_req_ready[lportid]),

            // AXI output
            .m_axis_tdata(fifo_lookup_req_index[lportid*KEY_SIZE +: KEY_SIZE]),
            .m_axis_tvalid(fifo_lookup_req_valid[lportid]),
            .m_axis_tready(fifo_lookup_req_ready[lportid])
        );

    end
endgenerate



axis_arb_mux #(
    .S_COUNT(LOOKUP_PORTS),
    .DATA_WIDTH(KEY_SIZE),
    .KEEP_ENABLE(0),
    .USER_ENABLE(1),
    .USER_WIDTH(LOOKUP_USER_WIDTH),
    .ARB_TYPE("ROUND_ROBIN")
)
lookup_mux (
    .clk(clk),
    .rst(rst),
    // AXI input
    .s_axis_tdata(fifo_lookup_req_index),
    .s_axis_tvalid(fifo_lookup_req_valid),
    .s_axis_tready(fifo_lookup_req_ready),
    .s_axis_tuser(lookup_req_user),

    // AXI output
    .m_axis_tdata(selected_lookup_req_index),
    .m_axis_tvalid(selected_lookup_req_valid),
    .m_axis_tready(selected_lookup_req_ready),
    .m_axis_tuser(selected_lookup_req_user)
);

cam_warpper #(
    .TABLE_SIZE(TABLE_SIZE),
    .KEY_SIZE(KEY_SIZE),
    .VALUE_SIZE(VALUE_SIZE),
    .UPDATE_USER_WIDTH(UPDATE_USER_WIDTH),
    .LOOKUP_USER_WIDTH(LOOKUP_USER_WIDTH)
)
cam_warpper (
    .clk(clk),
    .rst(rst),
    // AXI input
    .update_req_index(selected_update_req_indexdata[VALUE_SIZE +: KEY_SIZE]),
    .update_req_data(selected_update_req_indexdata[0 +: VALUE_SIZE]),
    .update_req_valid(selected_update_req_valid),
    .update_req_ready(selected_update_req_ready),
    .update_req_user(selected_update_req_user),

    // AXI output
    .lookup_req_index(selected_lookup_req_index),
    .lookup_req_valid(selected_lookup_req_valid),
    .lookup_req_ready(selected_lookup_req_ready),
    .lookup_req_user(selected_lookup_req_user),
    .lookup_value_data(lookup_result_data),
    .lookup_value_valid(lookup_result_valid),
    .lookup_value_ready(lookup_result_ready),
    .lookup_value_user(lookup_result_user)
);


logic [LOOKUP_PORTS-1:0]   lookup_result_ready_array;
assign lookup_result_ready = &lookup_result_ready_array;
genvar  demuxlportid;
generate
    for (demuxlportid=0; demuxlportid<LOOKUP_PORTS; demuxlportid = demuxlportid + 1) begin: lookup_demux

        logic  [VALUE_SIZE-1:0]                fifo_lookup_result_data;
        logic                                  fifo_lookup_result_valid;
        logic                                  fifo_lookup_result_ready;
        logic  [LOOKUP_USER_WIDTH-1:0]         fifo_lookup_result_user;

        axis_fifo #(
        .DEPTH(4),
        .DATA_WIDTH(VALUE_SIZE),
        .KEEP_ENABLE(0),
        .LAST_ENABLE(0),
        .ID_ENABLE(0),
        .DEST_ENABLE(0),
        .USER_ENABLE(0),
        .FRAME_FIFO(0)
        )
        lookupresult_fifo (
            .clk(clk),
            .rst(rst),
            // AXI input
            .s_axis_tdata({lookup_result_data}),
            .s_axis_tvalid(lookup_result_valid && (lookup_result_user == demuxlportid)),
            .s_axis_tready(lookup_result_ready_array[demuxlportid]),

            // AXI output
            .m_axis_tdata({fifo_lookup_result_data}),
            .m_axis_tvalid(fifo_lookup_result_valid),
            .m_axis_tready(fifo_lookup_result_ready)
        );

        always@* begin
            s_lookup_value_data[demuxlportid*VALUE_SIZE +: VALUE_SIZE] = fifo_lookup_result_data;
            s_lookup_value_valid[demuxlportid] = fifo_lookup_result_valid;
            fifo_lookup_result_ready = s_lookup_value_ready[demuxlportid];
        end
        
    end
endgenerate



endmodule

