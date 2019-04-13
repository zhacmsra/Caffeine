# Copyright 2014 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

################################################################
# This is a generated script based on design: opencldesign
#
################################################################


################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source opencldesign_script.tcl

# If you do not already have a project created,
# you can create a project using the following command:
#    create_project project_1 myproj -part xc7vx690tffg1157-2


# CHANGE DESIGN NAME HERE
set design_name opencldesign

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# CHECKING IF PROJECT EXISTS
if { [get_projects -quiet] eq "" } {
   puts "ERROR: Please open or create a project!"
   return 1
}


# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "ERROR: Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      puts "INFO: Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   puts "INFO: Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "ERROR: Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "ERROR: Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   puts "INFO: Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design $design_name

   puts "INFO: Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

puts "INFO: Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   puts $errMsg
   return $nRet
}

##################################################################
# DESIGN PROCs
##################################################################

# Hierarchical cell: OCL_REGION_0
proc create_hier_cell_OCL_REGION_0 { parentCell nameHier } {

  if { $parentCell eq "" || $nameHier eq "" } {
     puts "ERROR: create_hier_cell_OCL_REGION_0() - Empty argument(s)!"
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     puts "ERROR: Unable to find parent cell <$parentCell>!"
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     puts "ERROR: Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M00_AXI
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S00_AXI

  # Create pins
  create_bd_pin -dir I -type clk ACLK
  create_bd_pin -dir I -from 0 -to 0 -type rst ARESETN

  # Create instance: OCL_REGION_0, and set properties
  set OCL_REGION_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xcl_region:1.0 OCL_REGION_0 ]
  set_property -dict [ list CONFIG.NUM_MI {1}  ] $OCL_REGION_0

  # Create instance: axi_interconnect_0, and set properties
  set axi_interconnect_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0 ]
  set_property -dict [ list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}  ] $axi_interconnect_0

  # Create instance: axi_interconnect_1, and set properties
  set axi_interconnect_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1 ]
  set_property -dict [ list CONFIG.NUM_MI {1}  ] $axi_interconnect_1

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins S00_AXI] [get_bd_intf_pins axi_interconnect_1/S00_AXI]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins M00_AXI] [get_bd_intf_pins axi_interconnect_0/M00_AXI]
  connect_bd_intf_net -intf_net axi_interconnect_1_M00_AXI [get_bd_intf_pins OCL_REGION_0/S_AXI_CONTROL0] [get_bd_intf_pins axi_interconnect_1/M00_AXI]
  connect_bd_intf_net -intf_net xcl_region_0_M_AXI_GMEM0 [get_bd_intf_pins OCL_REGION_0/M_AXI_GMEM0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]

  # Create port connections
  connect_bd_net -net ACLK_1 [get_bd_pins ACLK] [get_bd_pins OCL_REGION_0/ACLK] [get_bd_pins axi_interconnect_0/ACLK] [get_bd_pins axi_interconnect_0/M00_ACLK] [get_bd_pins axi_interconnect_0/S00_ACLK] [get_bd_pins axi_interconnect_1/ACLK] [get_bd_pins axi_interconnect_1/M00_ACLK] [get_bd_pins axi_interconnect_1/S00_ACLK]
  connect_bd_net -net ARESETN_2 [get_bd_pins ARESETN] [get_bd_pins OCL_REGION_0/ARESETN] [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins axi_interconnect_0/M00_ARESETN] [get_bd_pins axi_interconnect_0/S00_ARESETN] [get_bd_pins axi_interconnect_1/ARESETN] [get_bd_pins axi_interconnect_1/M00_ARESETN] [get_bd_pins axi_interconnect_1/S00_ARESETN]

  # Restore current instance
  current_bd_instance $oldCurInst
}



# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     puts "ERROR: Unable to find parent cell <$parentCell>!"
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     puts "ERROR: Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports

  # Create ports

  # Create instance: c0_ddr_clk, and set properties
  set c0_ddr_clk [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_gen:1.0 c0_ddr_clk ]
  set_property -dict [ list CONFIG.INITIAL_RESET_CLOCK_CYCLES {5} CONFIG.FREQ_HZ {400000000} ] $c0_ddr_clk

  # Create instance: c0_ui_clk, and set properties
  set c0_ui_clk [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_gen:1.0 c0_ui_clk ]
  set_property -dict [ list CONFIG.INITIAL_RESET_CLOCK_CYCLES {5} CONFIG.FREQ_HZ {166666667} ] $c0_ui_clk

  # Create instance: OCL_REGION_0
  create_hier_cell_OCL_REGION_0 [current_bd_instance .] OCL_REGION_0

  # Create instance: sdaccel_generic_pcie_0, and set properties
  set sdaccel_generic_pcie_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:sdaccel_generic_pcie:1.0 sdaccel_generic_pcie_0 ]

  # Create interface connections
  connect_bd_intf_net -intf_net OCL_REGION_0_M00_AXI [get_bd_intf_pins OCL_REGION_0/M00_AXI] [get_bd_intf_pins sdaccel_generic_pcie_0/C0_DDR_SAXI]
  connect_bd_intf_net -intf_net sdaccel_generic_pcie_0_M_AXI_CTRL [get_bd_intf_pins OCL_REGION_0/S00_AXI] [get_bd_intf_pins sdaccel_generic_pcie_0/M_AXI_CTRL]

  # Create port connections
  connect_bd_net -net c0_ddr_clk_clk [get_bd_pins c0_ddr_clk/clk] [get_bd_pins sdaccel_generic_pcie_0/c0_ddr_clk]
  connect_bd_net -net c0_ui_clk_clk [get_bd_pins OCL_REGION_0/ACLK] [get_bd_pins c0_ui_clk/clk] [get_bd_pins sdaccel_generic_pcie_0/c0_ui_clk] [get_bd_pins sdaccel_generic_pcie_0/m_axi_ctrl_clk]
  connect_bd_net -net c0_ui_clk_sync_rst [get_bd_pins c0_ui_clk/sync_rst] [get_bd_pins OCL_REGION_0/ARESETN]

  # Create address segments
create_bd_addr_seg -range 0x40000000 -offset 0x0 [get_bd_addr_spaces sdaccel_generic_pcie_0/m_axi_ctrl] [get_bd_addr_segs OCL_REGION_0/OCL_REGION_0/S_AXI_CONTROL0_TERM/s_axi/reg0] SEG_OCL_REGION_0_reg0
  #create_bd_addr_seg -range 0x40000000 -offset 0x0 [get_bd_addr_spaces OCL_REGION_0/mmult/Data_m_axi_gmem] [get_bd_addr_segs hw_em_base_0/accel2xbar0/reg0] mmult_M_AXI_GMEM_OCL_REGION_0_M_AXI_GMEM0_reg0_0
 create_bd_addr_seg -range 0x40000000 -offset 0x0 [get_bd_addr_spaces OCL_REGION_0/OCL_REGION_0/M_AXI_GMEM0_TERM/m_axi] [get_bd_addr_segs sdaccel_generic_pcie_0/C0_DDR_SAXI/Reg] SEG_sdaccel_generic_pcie_0_reg0
 

  # Restore current instance
  current_bd_instance $oldCurInst

  save_bd_design
}
# End of create_root_design()

#############################
#Copied from m505-lx325 ocl region bd.tcl
proc gen_si_name      {idx} { return [format "S_AXI_CONTROL%d"      $idx ] }
proc gen_mi_name      {idx} { return [format "M_AXI_GMEM%d"      $idx ] }
proc gen_si_inst_name {idx} { return [format "S_AXI_CONTROL%d_TERM" $idx ] }
proc gen_mi_inst_name {idx} { return [format "M_AXI_GMEM%d_TERM" $idx ] }
proc gen_si_aclk_name {idx} { return [format "S_AXI_CONTROL%d_ACLK"     $idx ] }
proc gen_mi_aclk_name {idx} { return [format "M_AXI_GMEM%d_ACLK"     $idx ] }
proc gen_si_arst_name {idx} { return [format "S_AXI_CONTROL%d_ARESETN"  $idx ] }
proc gen_mi_arst_name {idx} { return [format "M_AXI_GMEM%d_ARESETN"  $idx ] }
proc gen_mi_cdatawidth_name {idx} {return [format "CONFIG.M%.2d_DATA_WIDTH" $idx]}
proc gen_si_cidwidth_name {idx} {return [format "CONFIG.S%.2d_ID_WIDTH" $idx]}
#############################

##################################################################
# MAIN FLOW
##################################################################

create_root_design ""
#set_param ips.enableSVCosim 1
#set_param project.allowSharedLibraryType 1
#set_property SELECTED_SIM_MODEL oclmdl_sim [get_ips opencldesign_hw_em_base_0_0]
#generate_target simulation [get_files  /proj/xhd_logicore_tools/users/sahilg/workspace/SDAccell/hw_em/exp/project_1.srcs/sources_1/bd/opencldesign/opencldesign.bd]

# XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
