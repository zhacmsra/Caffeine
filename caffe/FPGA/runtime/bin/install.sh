#!/bin/bash

# Deployment script for SDAccel compiled OpenCL programs

# This script will do the following:
# 1. Copy the necessary runtime files to the target directory specified with -d switch
# 2. Remove SDAccel's copy of libstdc++ and libOpenCL.so if the target system already
#    has the correct version of these files installed
# 3. Compile and install Linux kernel device drivers unless -k no switch is specifed
# 4. Install the firmware files like bitstreams and dsabins to the Linux firmware area
# 5. Generates a setup.sh environment script which may be used to setup correct environment
#    in order to run the compiled binaries.
#
# If run without root privileges, the script would not attempt to install firmware files
# or install the Linux kernel drivers


ROOT_DIR=`pwd`
DEST=
INSTALL_KERNEL_DRV="yes"

usage () {
    echo "Usage: $0 [-d <install_root>] [-k <yes|no>]"
    echo "       -d deployment directory"
    echo "       -k install kernel drivers"
    exit 1
}

while [[ $# -gt 0 ]]
do
    switch="$1"
    case $switch in
	-d)
	    DEST="$2"
	    shift
	    ;;
	-k)
	    INSTALL_KERNEL_DRV="$2"
	    shift
	    ;;
	-h)
	    usage
	    ;;
	*)
	    usage
	    ;;
    esac
    shift
done

PLAT=`uname -m`
OWNER=`whoami`

if [ -n "$DEST" ]; then
    mkdir -p $DEST > /dev/null 2>&1

    if [ $? -ne 0 ]; then
	echo "ERROR: $OWNER does not have write priviledges for $DEST"
	exit 1
    fi
else
    DEST=$ROOT_DIR
fi

# Create ICD entry for SDAccel if missing
if [ ! -e /etc/OpenCL/vendors/xilinx.icd ]; then
    if [ $OWNER != "root" ]; then
	echo "WARN: root priviledges required, skipping creating of ICD registry"
    else
	echo "INFO: Creating ICD entry for Xilinx Platform"
	echo "libxilinxopencl.so" > /etc/OpenCL/vendors/xilinx.icd
    fi
fi

# Check if an ICD is already installed then remove our copy of ICD
if [ -e /usr/lib/x86_64-linux-gnu/libOpenCL.so ] ||
       [ -e /usr/lib64/libOpenCL.so ]; then
    rm -f runtime/lib/$PLAT/libOpenCL.so*
fi

# Install the firmware to Linux firmware directory
if [ $OWNER != "root" ]; then
    echo "WARN: root priviledges required, skipping installation of firmware files"
else
    echo "INFO: Installing firmware for FPGA devices"
    install -d /lib/firmware/xilinx
    install -m 644 firmware/* /lib/firmware/xilinx
    if [ -e firmware/10ee-8038-0011-0000000000000000.dsabin ]; then
	rm -f /lib/firmware/xilinx/xilinx_adm-pcie-ku3_1ddr_2_0.dsabin
	# Support for legacy drivers which look for hardcoded dsabin names
	ln -s /lib/firmware/xilinx/10ee-8038-0011-0000000000000000.dsabin /lib/firmware/xilinx/xilinx_adm-pcie-ku3_1ddr_2_0.dsabin
    fi
fi

# Check if native C++ library supports C++ features we need
FOUND1=1
if [ -e /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
    nm -D /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.18 > /dev/null 2>&1
    FOUND1=$?
fi

FOUND2=1
if [ -e /usr/lib64/libstdc++.so.6 ]; then
    nm -D /usr/lib64/libstdc++.so.6 | grep GLIBCXX_3.4.18 > /dev/null 2>&1
    FOUND2=$?
fi

if [ $FOUND1 -eq 0 ] ||
       [ $FOUND2 -eq 0 ]; then
    rm -rf runtime/lib/$PLAT/libstdc++.so.6
fi

# Install runtime if a different destination folder is desired
if [ $ROOT_DIR != $DEST ]; then
    echo "INFO: Installing runtime libraries in $DEST"
    install -d $DEST/runtime/lib/$PLAT
    install runtime/lib/$PLAT/* $DEST/runtime/lib/$PLAT
fi

# Identify the preferred device
XCL_PLATFORM=`ls runtime/platforms | head -n 1`

# Compile and install Linux kernel drivers
mkdir -p /tmp/$$

index=0
for DEV in runtime/platforms/*; do
    cd $ROOT_DIR
    XCL_PLAT=`basename $DEV`;
    if [ $ROOT_DIR != $DEST ]; then
	echo "INFO: Device $XCL_PLAT"
	install -d $DEST/$DEV/driver
	install $DEV/driver/*.so $DEST/$DEV/driver
    fi
    if [ $INSTALL_KERNEL_DRV != "yes" ]; then
	continue;
    fi
    TEMP=/tmp/$$/$index
    rm -rf $TEMP
    mkdir -p $TEMP
    KERNEL_DRV_ZIP=`ls $DEV/driver/*.zip | head -n 1`
    cp $KERNEL_DRV_ZIP $TEMP
    cd $TEMP
    KERNEL_DRV_ZIP=`basename $KERNEL_DRV_ZIP`
    unzip $KERNEL_DRV_ZIP
    cd driver

    for DIR in *; do
	if [ ! -e $DIR/kernel/Makefile ]; then
	    continue
	fi
	cd $TEMP/driver/$DIR/kernel
	echo "INFO: building kernel driver"
	make
	MODULE=`ls *.ko | head -n 1`
	if [ $OWNER != "root" ]; then
	    echo "WARN: root priviledges required, skipping installation of kernel module $MODULE"
	else
	    MODULE=`basename $MODULE .ko`
	    make install
	    lsmod | grep $MODULE
	    if [ $? -eq 0 ] ; then
		rmmod $MODULE
	    fi
	    modprobe $MODULE
	fi
    done
    index=$((index + 1))
done

rm -rf /tmp/$$
cd $ROOT_DIR

echo "Generating SDAccel runtime environment setup script for bash, setup.sh"

echo "export XILINX_OPENCL=$DEST" > setup.sh
echo "export LD_LIBRARY_PATH=\$XILINX_OPENCL/runtime/lib/$PLAT:\$LD_LIBRARY_PATH" >> setup.sh
echo "export XCL_PLATFORM=$XCL_PLATFORM" >> setup.sh

echo "Generating SDAccel runtime environment setup script for tcsh, setup.csh"
echo "setenv XILINX_OPENCL $DEST" > setup.csh
echo "if ( ! \$?LD_LIBRARY_PATH ) then" >> setup.csh
echo "    setenv LD_LIBRARY_PATH \$XILINX_OPENCL/runtime/lib/$PLAT" >> setup.csh
echo "else" >> setup.csh
echo "    setenv LD_LIBRARY_PATH \$XILINX_OPENCL/runtime/lib/$PLAT:\$LD_LIBRARY_PATH" >> setup.csh
echo "endif" >> setup.csh
echo "setenv XCL_PLATFORM $XCL_PLATFORM" >> setup.csh

# XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
