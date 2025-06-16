// Generator : SpinalHDL v1.10.0    git head : 270018552577f3bb8e5339ee2583c9c22d324215
// Component : CAM
// Git hash  : 3a6304c32f02162848fb56753fa73afb0c4afdc2

`timescale 1ns/1ps

module CAM (
  input  wire          io_readRequest_valid,
  output wire          io_readRequest_ready,
  input  wire [7:0]    io_readRequest_payload,
  output wire          io_readResponse_valid,
  input  wire          io_readResponse_ready,
  output reg  [31:0]   io_readResponse_payload,
  input  wire          io_writeRequest_key_valid,
  output wire          io_writeRequest_key_ready,
  input  wire [7:0]    io_writeRequest_key_payload,
  input  wire          io_writeRequest_value_valid,
  output wire          io_writeRequest_value_ready,
  input  wire [31:0]   io_writeRequest_value_payload,
  input  wire          io_writeRequest_op_valid,
  output wire          io_writeRequest_op_ready,
  input  wire [7:0]    io_writeRequest_op_payload,
  input  wire          clk,
  input  wire          reset
);

  wire                io_writeRequest_key_fifo_io_flush;
  wire                io_writeRequest_value_fifo_io_flush;
  wire                io_writeRequest_op_fifo_io_flush;
  wire                io_readRequest_fifo_io_flush;
  wire                io_writeRequest_key_fifo_io_push_ready;
  wire                io_writeRequest_key_fifo_io_pop_valid;
  wire       [7:0]    io_writeRequest_key_fifo_io_pop_payload;
  wire       [4:0]    io_writeRequest_key_fifo_io_occupancy;
  wire       [4:0]    io_writeRequest_key_fifo_io_availability;
  wire                io_writeRequest_value_fifo_io_push_ready;
  wire                io_writeRequest_value_fifo_io_pop_valid;
  wire       [31:0]   io_writeRequest_value_fifo_io_pop_payload;
  wire       [4:0]    io_writeRequest_value_fifo_io_occupancy;
  wire       [4:0]    io_writeRequest_value_fifo_io_availability;
  wire                io_writeRequest_op_fifo_io_push_ready;
  wire                io_writeRequest_op_fifo_io_pop_valid;
  wire       [7:0]    io_writeRequest_op_fifo_io_pop_payload;
  wire       [4:0]    io_writeRequest_op_fifo_io_occupancy;
  wire       [4:0]    io_writeRequest_op_fifo_io_availability;
  wire                io_readRequest_fifo_io_push_ready;
  wire                io_readRequest_fifo_io_pop_valid;
  wire       [7:0]    io_readRequest_fifo_io_pop_payload;
  wire       [4:0]    io_readRequest_fifo_io_occupancy;
  wire       [4:0]    io_readRequest_fifo_io_availability;
  reg        [31:0]   _zz_io_readResponse_payload;
  wire       [1:0]    _zz_io_readResponse_payload_1;
  reg        [31:0]   _zz__zz_values_0;
  wire                writeCmd_valid;
  wire                writeCmd_ready;
  wire                writeCmd_fire;
  reg        [1:0]    next_1;
  reg        [7:0]    keys_0;
  reg        [7:0]    keys_1;
  reg        [7:0]    keys_2;
  reg        [7:0]    keys_3;
  reg        [31:0]   values_0;
  reg        [31:0]   values_1;
  reg        [31:0]   values_2;
  reg        [31:0]   values_3;
  wire                when_MyTopLevel_l65;
  wire       [1:0]    index;
  wire       [3:0]    _zz_1;
  wire       [3:0]    _zz_2;
  wire       [31:0]   _zz_values_0;
  wire       [3:0]    _zz_3;
  wire                _zz_4;
  wire                _zz_5;
  wire                _zz_6;
  wire                _zz_7;
  wire       [31:0]   _zz_values_0_1;
  wire       [31:0]   _zz_values_0_2;

  assign _zz_io_readResponse_payload_1 = ((keys_0 == io_readRequest_fifo_io_pop_payload) ? 2'b00 : ((keys_1 == io_readRequest_fifo_io_pop_payload) ? 2'b01 : ((keys_2 == io_readRequest_fifo_io_pop_payload) ? 2'b10 : 2'b11)));
  StreamFifo io_writeRequest_key_fifo (
    .io_push_valid   (io_writeRequest_key_valid                    ), //i
    .io_push_ready   (io_writeRequest_key_fifo_io_push_ready       ), //o
    .io_push_payload (io_writeRequest_key_payload[7:0]             ), //i
    .io_pop_valid    (io_writeRequest_key_fifo_io_pop_valid        ), //o
    .io_pop_ready    (writeCmd_fire                                ), //i
    .io_pop_payload  (io_writeRequest_key_fifo_io_pop_payload[7:0] ), //o
    .io_flush        (io_writeRequest_key_fifo_io_flush            ), //i
    .io_occupancy    (io_writeRequest_key_fifo_io_occupancy[4:0]   ), //o
    .io_availability (io_writeRequest_key_fifo_io_availability[4:0]), //o
    .clk             (clk                                          ), //i
    .reset           (reset                                        )  //i
  );
  StreamFifo_1 io_writeRequest_value_fifo (
    .io_push_valid   (io_writeRequest_value_valid                    ), //i
    .io_push_ready   (io_writeRequest_value_fifo_io_push_ready       ), //o
    .io_push_payload (io_writeRequest_value_payload[31:0]            ), //i
    .io_pop_valid    (io_writeRequest_value_fifo_io_pop_valid        ), //o
    .io_pop_ready    (writeCmd_fire                                  ), //i
    .io_pop_payload  (io_writeRequest_value_fifo_io_pop_payload[31:0]), //o
    .io_flush        (io_writeRequest_value_fifo_io_flush            ), //i
    .io_occupancy    (io_writeRequest_value_fifo_io_occupancy[4:0]   ), //o
    .io_availability (io_writeRequest_value_fifo_io_availability[4:0]), //o
    .clk             (clk                                            ), //i
    .reset           (reset                                          )  //i
  );
  StreamFifo io_writeRequest_op_fifo (
    .io_push_valid   (io_writeRequest_op_valid                    ), //i
    .io_push_ready   (io_writeRequest_op_fifo_io_push_ready       ), //o
    .io_push_payload (io_writeRequest_op_payload[7:0]             ), //i
    .io_pop_valid    (io_writeRequest_op_fifo_io_pop_valid        ), //o
    .io_pop_ready    (writeCmd_fire                               ), //i
    .io_pop_payload  (io_writeRequest_op_fifo_io_pop_payload[7:0] ), //o
    .io_flush        (io_writeRequest_op_fifo_io_flush            ), //i
    .io_occupancy    (io_writeRequest_op_fifo_io_occupancy[4:0]   ), //o
    .io_availability (io_writeRequest_op_fifo_io_availability[4:0]), //o
    .clk             (clk                                         ), //i
    .reset           (reset                                       )  //i
  );
  StreamFifo io_readRequest_fifo (
    .io_push_valid   (io_readRequest_valid                    ), //i
    .io_push_ready   (io_readRequest_fifo_io_push_ready       ), //o
    .io_push_payload (io_readRequest_payload[7:0]             ), //i
    .io_pop_valid    (io_readRequest_fifo_io_pop_valid        ), //o
    .io_pop_ready    (io_readResponse_ready                   ), //i
    .io_pop_payload  (io_readRequest_fifo_io_pop_payload[7:0] ), //o
    .io_flush        (io_readRequest_fifo_io_flush            ), //i
    .io_occupancy    (io_readRequest_fifo_io_occupancy[4:0]   ), //o
    .io_availability (io_readRequest_fifo_io_availability[4:0]), //o
    .clk             (clk                                     ), //i
    .reset           (reset                                   )  //i
  );
  always @(*) begin
    case(_zz_io_readResponse_payload_1)
      2'b00 : _zz_io_readResponse_payload = values_0;
      2'b01 : _zz_io_readResponse_payload = values_1;
      2'b10 : _zz_io_readResponse_payload = values_2;
      default : _zz_io_readResponse_payload = values_3;
    endcase
  end

  always @(*) begin
    case(index)
      2'b00 : _zz__zz_values_0 = values_0;
      2'b01 : _zz__zz_values_0 = values_1;
      2'b10 : _zz__zz_values_0 = values_2;
      default : _zz__zz_values_0 = values_3;
    endcase
  end

  assign io_writeRequest_key_ready = io_writeRequest_key_fifo_io_push_ready;
  assign io_writeRequest_value_ready = io_writeRequest_value_fifo_io_push_ready;
  assign io_writeRequest_op_ready = io_writeRequest_op_fifo_io_push_ready;
  assign writeCmd_fire = (writeCmd_valid && writeCmd_ready);
  assign writeCmd_valid = ((io_writeRequest_key_fifo_io_pop_valid && io_writeRequest_value_fifo_io_pop_valid) && io_writeRequest_op_fifo_io_pop_valid);
  assign writeCmd_ready = 1'b1;
  assign io_readRequest_ready = io_readRequest_fifo_io_push_ready;
  assign io_readResponse_valid = io_readRequest_fifo_io_pop_valid;
  assign when_MyTopLevel_l65 = (writeCmd_fire && (io_writeRequest_key_payload == io_readRequest_fifo_io_pop_payload));
  always @(*) begin
    if(when_MyTopLevel_l65) begin
      io_readResponse_payload = io_writeRequest_value_payload;
    end else begin
      io_readResponse_payload = _zz_io_readResponse_payload;
    end
  end

  assign index = ((keys_0 == io_writeRequest_key_payload) ? 2'b00 : ((keys_1 == io_writeRequest_key_payload) ? 2'b01 : ((keys_2 == io_writeRequest_key_payload) ? 2'b10 : 2'b11)));
  assign _zz_1 = ({3'd0,1'b1} <<< next_1);
  assign _zz_2 = ({3'd0,1'b1} <<< next_1);
  assign _zz_values_0 = _zz__zz_values_0;
  assign _zz_3 = ({3'd0,1'b1} <<< index);
  assign _zz_4 = _zz_3[0];
  assign _zz_5 = _zz_3[1];
  assign _zz_6 = _zz_3[2];
  assign _zz_7 = _zz_3[3];
  assign _zz_values_0_1 = (_zz_values_0 + io_writeRequest_value_payload);
  assign _zz_values_0_2 = (_zz_values_0 - io_writeRequest_value_payload);
  assign io_writeRequest_key_fifo_io_flush = 1'b0;
  assign io_writeRequest_value_fifo_io_flush = 1'b0;
  assign io_writeRequest_op_fifo_io_flush = 1'b0;
  assign io_readRequest_fifo_io_flush = 1'b0;
  always @(posedge clk or posedge reset) begin
    if(reset) begin
      next_1 <= 2'b00;
      keys_0 <= 8'h00;
      keys_1 <= 8'h00;
      keys_2 <= 8'h00;
      keys_3 <= 8'h00;
      values_0 <= 32'h00000000;
      values_1 <= 32'h00000000;
      values_2 <= 32'h00000000;
      values_3 <= 32'h00000000;
    end else begin
      if(writeCmd_fire) begin
        case(io_writeRequest_op_payload)
          8'h00 : begin
            if(_zz_1[0]) begin
              keys_0 <= io_writeRequest_key_payload;
            end
            if(_zz_1[1]) begin
              keys_1 <= io_writeRequest_key_payload;
            end
            if(_zz_1[2]) begin
              keys_2 <= io_writeRequest_key_payload;
            end
            if(_zz_1[3]) begin
              keys_3 <= io_writeRequest_key_payload;
            end
            if(_zz_2[0]) begin
              values_0 <= io_writeRequest_value_payload;
            end
            if(_zz_2[1]) begin
              values_1 <= io_writeRequest_value_payload;
            end
            if(_zz_2[2]) begin
              values_2 <= io_writeRequest_value_payload;
            end
            if(_zz_2[3]) begin
              values_3 <= io_writeRequest_value_payload;
            end
            next_1 <= (next_1 + 2'b01);
          end
          8'h01 : begin
            if(_zz_4) begin
              values_0 <= _zz_values_0_1;
            end
            if(_zz_5) begin
              values_1 <= _zz_values_0_1;
            end
            if(_zz_6) begin
              values_2 <= _zz_values_0_1;
            end
            if(_zz_7) begin
              values_3 <= _zz_values_0_1;
            end
          end
          8'h02 : begin
            if(_zz_4) begin
              values_0 <= _zz_values_0_2;
            end
            if(_zz_5) begin
              values_1 <= _zz_values_0_2;
            end
            if(_zz_6) begin
              values_2 <= _zz_values_0_2;
            end
            if(_zz_7) begin
              values_3 <= _zz_values_0_2;
            end
          end
          default : begin
          end
        endcase
      end
    end
  end


endmodule

//StreamFifo_3 replaced by StreamFifo

//StreamFifo_2 replaced by StreamFifo

module StreamFifo_1 (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [31:0]   io_push_payload,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [31:0]   io_pop_payload,
  input  wire          io_flush,
  output wire [4:0]    io_occupancy,
  output wire [4:0]    io_availability,
  input  wire          clk,
  input  wire          reset
);

  reg        [31:0]   _zz_logic_ram_port1;
  wire       [31:0]   _zz_logic_ram_port;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [4:0]    logic_ptr_push;
  reg        [4:0]    logic_ptr_pop;
  wire       [4:0]    logic_ptr_occupancy;
  wire       [4:0]    logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                io_push_fire;
  wire                logic_push_onRam_write_valid;
  wire       [3:0]    logic_push_onRam_write_payload_address;
  wire       [31:0]   logic_push_onRam_write_payload_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [3:0]    logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [3:0]    logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [3:0]    logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [3:0]    logic_pop_sync_readPort_cmd_payload;
  wire       [31:0]   logic_pop_sync_readPort_rsp;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [31:0]   logic_pop_sync_readArbitation_translated_payload;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [4:0]    logic_pop_sync_popReg;
  reg [31:0] logic_ram [0:15];

  assign _zz_logic_ram_port = logic_push_onRam_write_payload_data;
  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= _zz_logic_ram_port;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = (((logic_ptr_push ^ logic_ptr_popOnIo) ^ 5'h10) == 5'h00);
  assign logic_ptr_empty = (logic_ptr_push == logic_ptr_pop);
  assign logic_ptr_occupancy = (logic_ptr_push - logic_ptr_popOnIo);
  assign io_push_ready = (! logic_ptr_full);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push[3:0];
  assign logic_push_onRam_write_payload_data = io_push_payload;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop[3:0];
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp = _zz_logic_ram_port1;
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload = logic_pop_sync_readPort_rsp;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload = logic_pop_sync_readArbitation_translated_payload;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (5'h10 - logic_ptr_occupancy);
  always @(posedge clk or posedge reset) begin
    if(reset) begin
      logic_ptr_push <= 5'h00;
      logic_ptr_pop <= 5'h00;
      logic_ptr_wentUp <= 1'b0;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 5'h00;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 5'h01);
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 5'h01);
      end
      if(io_flush) begin
        logic_ptr_push <= 5'h00;
        logic_ptr_pop <= 5'h00;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 5'h00;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

module StreamFifo (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [7:0]    io_push_payload,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [7:0]    io_pop_payload,
  input  wire          io_flush,
  output wire [4:0]    io_occupancy,
  output wire [4:0]    io_availability,
  input  wire          clk,
  input  wire          reset
);

  reg        [7:0]    _zz_logic_ram_port1;
  wire       [7:0]    _zz_logic_ram_port;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [4:0]    logic_ptr_push;
  reg        [4:0]    logic_ptr_pop;
  wire       [4:0]    logic_ptr_occupancy;
  wire       [4:0]    logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                io_push_fire;
  wire                logic_push_onRam_write_valid;
  wire       [3:0]    logic_push_onRam_write_payload_address;
  wire       [7:0]    logic_push_onRam_write_payload_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [3:0]    logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [3:0]    logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [3:0]    logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [3:0]    logic_pop_sync_readPort_cmd_payload;
  wire       [7:0]    logic_pop_sync_readPort_rsp;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [7:0]    logic_pop_sync_readArbitation_translated_payload;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [4:0]    logic_pop_sync_popReg;
  reg [7:0] logic_ram [0:15];

  assign _zz_logic_ram_port = logic_push_onRam_write_payload_data;
  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= _zz_logic_ram_port;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = (((logic_ptr_push ^ logic_ptr_popOnIo) ^ 5'h10) == 5'h00);
  assign logic_ptr_empty = (logic_ptr_push == logic_ptr_pop);
  assign logic_ptr_occupancy = (logic_ptr_push - logic_ptr_popOnIo);
  assign io_push_ready = (! logic_ptr_full);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push[3:0];
  assign logic_push_onRam_write_payload_data = io_push_payload;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop[3:0];
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp = _zz_logic_ram_port1;
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload = logic_pop_sync_readPort_rsp;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload = logic_pop_sync_readArbitation_translated_payload;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (5'h10 - logic_ptr_occupancy);
  always @(posedge clk or posedge reset) begin
    if(reset) begin
      logic_ptr_push <= 5'h00;
      logic_ptr_pop <= 5'h00;
      logic_ptr_wentUp <= 1'b0;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 5'h00;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 5'h01);
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 5'h01);
      end
      if(io_flush) begin
        logic_ptr_push <= 5'h00;
        logic_ptr_pop <= 5'h00;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 5'h00;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

