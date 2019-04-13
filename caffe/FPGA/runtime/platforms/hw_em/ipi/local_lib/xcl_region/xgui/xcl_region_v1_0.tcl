proc gen_param_table { ipview  parent table_name table_prefix num_rows num_columns tl_text lcolumn_prefix param_prefix widget_order {j_iter 1}} {

  set table [ipgui::add_table $ipview -name $table_name -rows $num_rows -columns $num_columns -parent $parent]

  set tl [ipgui::add_static_text $ipview -name ${table_prefix}_HEADER_0 -parent $table -text $tl_text]
  set_property cell_location 0,0 $tl

  for {set i 0} {$i < 16} {incr i } {
    set param [ipgui::add_static_text $ipview -name ${table_prefix}${i}_LHS -parent $table -text "[subst $lcolumn_prefix]"]
    set_property cell_location [expr $i + 1],0 $param

    set column_num 1
    for {set j 0} {$j < $j_iter} {incr j} {
      foreach {column_title param_root widget_type} $widget_order {

        set actual_column_title [subst $column_title]

        if {![info exists headers($actual_column_title)]} {
          set header [ipgui::add_static_text $ipview -name ${table_prefix}_HEADER_$i -parent $table -text $actual_column_title]
          set_property cell_location 0,$column_num $header
          array set headers [list $actual_column_title true]
        }

        set param_name [subst $param_prefix]_$param_root
        set param [ipgui::add_param $ipview -parent $table -widget $widget_type -name $param_name]
        set_property cell_location [expr $i+1],$column_num $param
        incr column_num
      }
    }
  }
}

proc create_gui { ipview } {
	set Page0 [ ipgui::add_page $ipview  -name "Interface Selection" -layout vertical]
	set Component_Name [ ipgui::add_param  $ipview  -parent  $Page0  -name Component_Name ]
	set NUM_SI    [ipgui::add_param $ipview -parent $Page0 -name NUM_SI -widget comboBox ]
	set NUM_MI    [ipgui::add_param $ipview -parent $Page0 -name NUM_MI -widget comboBox ]	

  set MI_CONFIG [ ipgui::add_page $ipview -name "Master Interfaces" -layout vertical ]
  set widget_order { "DataWidth" DATA_WIDTH comboBox }
  gen_param_table $ipview $MI_CONFIG MI_TABLE master_settings_table_ 17 2 "Master Interface" {M[format %.2d $i]_AXI} {M[format %.2d $i]} $widget_order

  set SI_CONFIG [ ipgui::add_page $ipview -name "Slave Interfaces" -layout vertical ]
  set widget_order { "ID Width" ID_WIDTH comboBox }
  gen_param_table $ipview $SI_CONFIG SI_TABLE slave_settings_table_ 17 2 "Slave Interface" {S[format %.2d $i]_AXI} {S[format %.2d $i]} $widget_order
}

proc NUM_MI_updated { ipview } {
  set NUM_MI [get_param_value NUM_MI]
  set hidden_rows ""
  for {set i [expr $NUM_MI+1]} {$i < 17} {incr i} {
    set hidden_rows [join [lappend hidden_rows $i] ,]
  }
  set_property hidden_rows "$hidden_rows" [ipgui::get_tablespec "MI_TABLE" -of $ipview]
}

proc NUM_SI_updated { ipview } {
  set NUM_SI [get_param_value NUM_SI]
  set hidden_rows ""
  for {set i [expr $NUM_SI+1]} {$i < 17} {incr i} {
    set hidden_rows [join [lappend hidden_rows $i] ,]
  }
  set_property hidden_rows "$hidden_rows" [ipgui::get_tablespec "SI_TABLE" -of $ipview]
}


# XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
