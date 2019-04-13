# Define the solution for SDAccel
#create_solution -name myproj_resyn_8_xfcn -dir . -force
create_solution -name myproj_resyn_8 -dir . -force
#add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.1
add_device -vbnv xilinx:adm-pcie-ku3:2ddr:2.1

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE -D C_KERNEL" -objects [current_solution]

# Host Source Files
# add_files "kernel-cl.cpp"
#add_files "bwt_acc.h" 

# Kernel Definition
create_kernel vgg16 -type c

#add_files -kernel [get_kernels vgg16] "falconML.hpp"
add_files -kernel [get_kernels vgg16] "vgg16.cpp"

# Define Binary Containers
create_opencl_binary vgg16
set_property region "OCL_REGION_0" [get_opencl_binary vgg16]
create_compute_unit -opencl_binary [get_opencl_binary vgg16] -kernel [get_kernels vgg16] -name k1
set_param compiler.worstNegativeSlack -1.0

# Compile the design for CPU based emulation
# compile_emulation -flow cpu -opencl_binary [get_opencl_binary vgg16]

# Run the compiled application in CPU based emulation mode
# run_emulation -flow cpu -args "vgg16.xclbin /home/peichen/work/test/genome/bwt/input_small.txt"
# run_emulation -flow cpu -args "vgg16.xclbin"

#exit

# Compile the design for hardware based emulation
# compile_emulation -flow hardware -opencl_binary [get_opencl_binary vgg16]

# Run the compiled application in hardware based emulation mode
# run_emulation -flow hardware -args "vgg16.xclbin /home/peichen/work/test/genome/bwt/input_small.txt"

# report_estimate

# Compile the application to run on the accelerator card
build_system

# Package the application binaries
# package_system

#run_system -args "vgg16.xclbin"

