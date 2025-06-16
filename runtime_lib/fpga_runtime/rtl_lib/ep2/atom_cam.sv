// Generator : SpinalHDL v1.10.0    git head : 270018552577f3bb8e5339ee2583c9c22d324215
// Component : CAM
// Git hash  : 3a6304c32f02162848fb56753fa73afb0c4afdc2

`timescale 1ns/1ps

module atom_CAM (
  input  wire          s_lookup_req_valid,
  output wire          s_lookup_req_ready,
  input  wire [7:0]    s_lookup_req_index,
  output wire          s_lookup_value_valid,
  input  wire          s_lookup_value_ready,
  output reg  [31:0]   s_lookup_value_data,
  input  wire          s_update_req_index_valid,
  output wire          s_update_req_index_ready,
  input  wire [7:0]    s_update_req_index,
  input  wire          s_update_req_data_valid,
  output wire          s_update_req_data_ready,
  input  wire [31:0]   s_update_req_data,
  input  wire          clk,
  input  wire          rst
);

CAM #() CAMinst (
  .io_readRequest_valid(s_lookup_req_valid),
  .io_readRequest_ready(s_lookup_req_ready),
  .io_readRequest_payload(s_lookup_req_index),
  .io_readResponse_valid(s_lookup_value_valid),
  .io_readResponse_ready(s_lookup_value_ready),
  .io_readResponse_payload(s_lookup_value_data),
  .io_writeRequest_key_valid(s_update_req_index_valid),
  .io_writeRequest_key_ready(s_update_req_index_ready),
  .io_writeRequest_key_payload(s_update_req_index),
  .io_writeRequest_value_valid(s_update_req_data_valid),
  .io_writeRequest_value_ready(s_update_req_data_ready),
  .io_writeRequest_value_payload(s_update_req_data),
  .io_writeRequest_op_valid(1),
  .io_writeRequest_op_ready(),
  .io_writeRequest_op_payload(1),
  .clk(clk),
  .reset(rst)
);

endmodule

