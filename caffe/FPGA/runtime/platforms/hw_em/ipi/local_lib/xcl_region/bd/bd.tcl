proc init { cell_name args } {
  return []
}
proc pre_propagate { this {prop_info {}} } {
}
proc post_propagate { this {prop_info {}} } {
}
proc gen_si_inst_name {idx} { return [format "S_AXI_CONTROL%d_TERM" $idx ] }
proc gen_mi_inst_name {idx} { return [format "M_AXI_GMEM%d_TERM" $idx ] }
proc gen_si_name      {idx} { return [format "S_AXI_CONTROL%d"      $idx ] }
proc gen_mi_name      {idx} { return [format "M_AXI_GMEM%d"      $idx ] }
proc gen_si_aclk_name {idx} { return [format "S_AXI_CONTROL%d_ACLK"     $idx ] }
proc gen_mi_aclk_name {idx} { return [format "M_AXI_GMEM%d_ACLK"     $idx ] }
proc gen_si_arst_name {idx} { return [format "S_AXI_CONTROL%d_ARESETN"  $idx ] }
proc gen_mi_arst_name {idx} { return [format "M_AXI_GMEM%d_ARESETN"  $idx ] }

#proc gen_si_inst_name {idx} { return [format "S%.2d_AXI_TERM" $idx ] }
#proc gen_mi_inst_name {idx} { return [format "M%.2d_AXI_TERM" $idx ] }
#proc gen_si_name      {idx} { return [format "S%.2d_AXI"      $idx ] }
#proc gen_mi_name      {idx} { return [format "M%.2d_AXI"      $idx ] }
#proc gen_si_aclk_name {idx} { return [format "S%.2d_ACLK"     $idx ] }
#proc gen_mi_aclk_name {idx} { return [format "M%.2d_ACLK"     $idx ] }
#proc gen_si_arst_name {idx} { return [format "S%.2d_ARESETN"  $idx ] }
#proc gen_mi_arst_name {idx} { return [format "M%.2d_ARESETN"  $idx ] }

proc gen_mi_cdatawidth_name {idx} {return [format "CONFIG.M%.2d_DATA_WIDTH" $idx]}
proc gen_si_cidwidth_name {idx} {return [format "CONFIG.S%.2d_ID_WIDTH" $idx]}

proc update_boundary { this {prop_info {}} } {
  global env

  if {[info exists env(XIL_IFX_DISABLE_APPCORE_UPDATE_BOUNDARY)]} {
    return
  }

  set obj [get_bd_cells $this]
  current_bd_instance $obj

  set si_pins [get_bd_intf_pins -hier -regexp "S_AXI_CONTROL\[0-9\]*"]
  set mi_pins [get_bd_intf_pins -hier -regexp "M_AXI_GMEM\[0-9\]*"]
#  set si_pins [get_bd_intf_pins -hier -regexp "S\[0-9\]*_AXI"]
#  set mi_pins [get_bd_intf_pins -hier -regexp "M\[0-9\]*_AXI"]
  set num_si_pins [llength $si_pins]
  set num_mi_pins [llength $mi_pins]

  set num_si [get_property CONFIG.NUM_SI $obj]
  set num_mi [get_property CONFIG.NUM_MI $obj]

  if {[get_bd_pins $obj/ACLK] == ""} {
    set ict_clk [create_bd_pin -type clk -dir I ACLK]
  } else {
    set ict_clk [get_bd_pins $obj/ACLK]
  }

  if {[get_bd_pins $obj/ARESETN] == ""} {
    set ict_rst [create_bd_pin -type rst -dir I ARESETN]
    set_property CONFIG.ASSOCIATED_RESET ARESETN $ict_clk
  } else {
    set ict_rst [get_bd_pins $obj/ARESETN]
  }

  if { $num_si > $num_si_pins } {
    for {set i $num_si_pins} {$i < $num_si} {incr i} {
      set intf [create_bd_intf_pin -vlnv xilinx.com:interface:aximm_rtl:1.0 -mode slave [gen_si_name $i]]
      set abusif [get_property CONFIG.ASSOCIATED_BUSIF $ict_clk]
      if { [string compare $abusif ""] == 0 } {
        set_property CONFIG.ASSOCIATED_BUSIF "[gen_si_name $i]" $ict_clk
      } else {
        set_property CONFIG.ASSOCIATED_BUSIF "$abusif:[gen_si_name $i]" $ict_clk
      }
    }
  }
  
  if { $num_si < $num_si_pins } {
    for {set i $num_si} {$i < $num_si_pins} {incr i} {
      delete_bd_objs [get_bd_intf_pins [gen_si_name $i]]
    }
  }
  if { $num_mi > $num_mi_pins } {
    for {set i $num_mi_pins} {$i < $num_mi} {incr i} {
      set intf [create_bd_intf_pin -vlnv xilinx.com:interface:aximm_rtl:1.0 -mode master [gen_mi_name $i]]
      set abusif [get_property CONFIG.ASSOCIATED_BUSIF $ict_clk]
      if { [string compare $abusif ""] == 0 } {
        set_property CONFIG.ASSOCIATED_BUSIF "[gen_mi_name $i]" $ict_clk
      } else {
        set_property CONFIG.ASSOCIATED_BUSIF "$abusif:[gen_mi_name $i]" $ict_clk
      }
    }
  }
  
  if { $num_mi < $num_mi_pins } {
    for {set i $num_mi} {$i < $num_mi_pins} {incr i} {
      delete_bd_objs [get_bd_intf_pins [gen_mi_name $i]]
    }
  }

}
proc update_contents {this {prop_info {}} } {

  if {[info exists env(XIL_IFX_DISABLE_APPCORE_UPDATE_CONTENTS)]} {
    return
  }

  set obj [get_bd_cells $this]
  current_bd_instance $obj

  set num_si [get_property CONFIG.NUM_SI $obj]
  set num_mi [get_property CONFIG.NUM_MI $obj]    

  set ict_clk [get_bd_pins $obj/ACLK]
  set ict_rst [get_bd_pins $obj/ARESETN]


  # Clean up contents
  foreach obj_type {nets intf_nets} {
    set objs [get_bd_$obj_type]
    if {$objs != ""} {
      delete_bd_objs $objs
    }
  }
  delete_bd_objs [get_bd_cells] 

  # Create Global Clock / Reset
  set ict_clk_net [create_bd_net [lindex [split $obj /] end]_ACLK_net]
  set ict_rst_net [create_bd_net [lindex [split $obj /] end]_ARESETN_net]
  connect_bd_net -net $ict_clk_net $ict_clk
  connect_bd_net -net $ict_rst_net $ict_rst

  # Generate Master Instances
  for {set i 0} {$i < $num_mi} {incr i} {
    set inst [create_bd_cell -vlnv xilinx.com:ip:axi_master_term:1.0 -type ip -name [gen_mi_inst_name $i]]
    set_property CONFIG.C_AXI_DATA_WIDTH [get_property [gen_mi_cdatawidth_name $i] $obj] $inst
    connect_bd_intf_net [get_bd_intf_pins [gen_mi_name $i]] [get_bd_intf_pins $inst/m_axi]
  }

  # Generate Slave Instances
  for {set i 0} {$i < $num_si} {incr i} {
    set inst [create_bd_cell -vlnv xilinx.com:ip:axi_slave_term:1.0 -type ip -name [gen_si_inst_name $i]]
    set_property CONFIG.C_AXI_ID_WIDTH [get_property [gen_si_cidwidth_name $i] $obj] $inst
    set_property CONFIG.C_AXI_PROTOCOL "AXI4LITE" $inst
    connect_bd_intf_net [get_bd_intf_pins [gen_si_name $i]] [get_bd_intf_pins $inst/s_axi]
  }

  # Connect ACLK/ARESETN
  connect_bd_net -net $ict_clk_net [get_bd_pins -hier -regexp "\.\*_term\.\*axi_aclk"]
  connect_bd_net -net $ict_rst_net [get_bd_pins -hier -regexp "\.\*_term\.\*axi_aresetn"]

}

# XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
