#Definitional proc to organize widgets for parameters.
proc create_gui { ipview } {
	set Page0 [ ipgui::add_page $ipview  -name "Page 0" -layout vertical]
	set Component_Name [ ipgui::add_param  $ipview  -parent  $Page0  -name Component_Name ]
	set C_AXI_PROTOCOL [ipgui::add_param $ipview -parent $Page0 -name C_AXI_PROTOCOL]
	set C_AXI_ID_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_ID_WIDTH]
	set C_AXI_ADDR_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_ADDR_WIDTH]
	set C_AXI_DATA_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_DATA_WIDTH]
	set C_AXI_AWUSER_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_AWUSER_WIDTH]
	set C_AXI_WUSER_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_WUSER_WIDTH]
	set C_AXI_BUSER_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_BUSER_WIDTH]
	set C_AXI_ARUSER_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_ARUSER_WIDTH]
	set C_AXI_RUSER_WIDTH [ipgui::add_param $ipview -parent $Page0 -name C_AXI_RUSER_WIDTH]
}

proc C_AXI_PROTOCOL_updated {ipview} {
	# Procedure called when C_AXI_PROTOCOL is updated
	return true
}

proc validate_C_AXI_PROTOCOL {ipview} {
	# Procedure called to validate C_AXI_PROTOCOL
	return true
}

proc C_AXI_ID_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_ID_WIDTH is updated
	return true
}

proc validate_C_AXI_ID_WIDTH {ipview} {
	# Procedure called to validate C_AXI_ID_WIDTH
	return true
}

proc C_AXI_ADDR_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_ADDR_WIDTH is updated
	return true
}

proc validate_C_AXI_ADDR_WIDTH {ipview} {
	# Procedure called to validate C_AXI_ADDR_WIDTH
	return true
}

proc C_AXI_DATA_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_DATA_WIDTH is updated
	return true
}

proc validate_C_AXI_DATA_WIDTH {ipview} {
	# Procedure called to validate C_AXI_DATA_WIDTH
	return true
}

proc C_AXI_AWUSER_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_AWUSER_WIDTH is updated
	return true
}

proc validate_C_AXI_AWUSER_WIDTH {ipview} {
	# Procedure called to validate C_AXI_AWUSER_WIDTH
	return true
}

proc C_AXI_WUSER_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_WUSER_WIDTH is updated
	return true
}

proc validate_C_AXI_WUSER_WIDTH {ipview} {
	# Procedure called to validate C_AXI_WUSER_WIDTH
	return true
}

proc C_AXI_BUSER_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_BUSER_WIDTH is updated
	return true
}

proc validate_C_AXI_BUSER_WIDTH {ipview} {
	# Procedure called to validate C_AXI_BUSER_WIDTH
	return true
}

proc C_AXI_ARUSER_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_ARUSER_WIDTH is updated
	return true
}

proc validate_C_AXI_ARUSER_WIDTH {ipview} {
	# Procedure called to validate C_AXI_ARUSER_WIDTH
	return true
}

proc C_AXI_RUSER_WIDTH_updated {ipview} {
	# Procedure called when C_AXI_RUSER_WIDTH is updated
	return true
}

proc validate_C_AXI_RUSER_WIDTH {ipview} {
	# Procedure called to validate C_AXI_RUSER_WIDTH
	return true
}


proc updateModel_C_AXI_PROTOCOL {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_PROTOCOL -of $ipview ]] [ipgui::get_modelparamspec C_AXI_PROTOCOL -of $ipview ]

	return true
}

proc updateModel_C_AXI_ID_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_ID_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_ID_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_ADDR_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_ADDR_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_ADDR_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_DATA_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_DATA_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_DATA_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_AWUSER_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_AWUSER_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_AWUSER_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_WUSER_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_WUSER_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_WUSER_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_BUSER_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_BUSER_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_BUSER_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_ARUSER_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_ARUSER_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_ARUSER_WIDTH -of $ipview ]

	return true
}

proc updateModel_C_AXI_RUSER_WIDTH {ipview} {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

	set_property modelparam_value [get_property value [ipgui::get_paramspec C_AXI_RUSER_WIDTH -of $ipview ]] [ipgui::get_modelparamspec C_AXI_RUSER_WIDTH -of $ipview ]

	return true
}


# XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
