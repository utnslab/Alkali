
set_property -dict {LOC AR15} [get_ports refclk_1_p] ;
set_property -dict {LOC AR14} [get_ports refclk_1_n] ;
set_property -dict {LOC BH26 IOSTANDARD LVCMOS18 PULLUP true} [get_ports reset_n] ;


create_clock -period 4 -name mgt_refclk_1 [get_ports refclk_1_p]

set_false_path -from [get_ports {pcie_reset_n}]
set_input_delay 0 [get_ports {pcie_reset_n}]