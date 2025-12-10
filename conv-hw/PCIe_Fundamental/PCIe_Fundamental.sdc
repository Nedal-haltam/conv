#**************************************************************
# This .sdc file is created by Terasic Tool.
# Users are recommended to modify this file to match users logic.
#**************************************************************

#**************************************************************
# Create Clock
#**************************************************************
create_clock -period "50.000000 MHz" [get_ports CLOCK_50_B3B]
create_clock -period "50.000000 MHz" [get_ports CLOCK_50_B4A]
create_clock -period "50.000000 MHz" [get_ports CLOCK_50_B5B]
create_clock -period "50.000000 MHz" [get_ports CLOCK_50_B6A]
create_clock -period "50.000000 MHz" [get_ports CLOCK_50_B7A]
create_clock -period "50.000000 MHz" [get_ports CLOCK_50_B8A]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]

#**************************************************************
# Create Generated Clock
#**************************************************************
derive_pll_clocks



#**************************************************************
# Set Clock Latency
#**************************************************************



#**************************************************************
# Set Clock Uncertainty
#**************************************************************
derive_clock_uncertainty



#**************************************************************
# Set Input Delay
#**************************************************************



#**************************************************************
# Set Output Delay
#**************************************************************



#**************************************************************
# Set Clock Groups
#**************************************************************
set_clock_groups -asynchronous -group {[get_clocks {CLOCK_50_B3B}]}
set_clock_groups -asynchronous -group {[get_clocks {PCIE_REFCLK_p}]}


#**************************************************************
# Set False Path
#**************************************************************
set_false_path -from [get_ports {PCIE_PERST_n}]
set_false_path -from [get_ports {CPU_RESET_n}]
set_false_path -from [get_ports {KEY[*]}]
set_false_path -to   [get_ports {LED[*]}]

set_false_path -from {any_rstn_rr}
set_false_path -from [get_clocks {CLOCK_50_B3B}] -to [get_clocks {*arriav_hd_altpe2_hip_top|coreclkout}]


#**************************************************************
# Set Multicycle Path
#**************************************************************



#**************************************************************
# Set Maximum Delay
#**************************************************************



#**************************************************************
# Set Minimum Delay
#**************************************************************



#**************************************************************
# Set Input Transition
#**************************************************************



#**************************************************************
# Set Load
#**************************************************************



