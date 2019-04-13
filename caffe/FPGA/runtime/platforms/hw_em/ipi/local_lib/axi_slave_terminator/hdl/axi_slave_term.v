//(c) Copyright 2013 Xilinx, Inc. All rights reserved.
//
//  This file contains confidential and proprietary information
//  of Xilinx, Inc. and is protected under U.S. and
//  international copyright and other intellectual property
//  laws.
//
//  DISCLAIMER
//  This disclaimer is not a license and does not grant any
//  rights to the materials distributed herewith. Except as
//  otherwise provided in a valid license issued to you by
//  Xilinx, and to the maximum extent permitted by applicable
//  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
//  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
//  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
//  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
//  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
//  (2) Xilinx shall not be liable (whether in contract or tort,
//  including negligence, or under any other theory of
//  liability) for any loss or damage of any kind or nature
//  related to, arising under or in connection with these
//  materials, including for any direct, or any indirect,
//  special, incidental, or consequential loss or damage
//  (including loss of data, profits, goodwill, or any type of
//  loss or damage suffered as a result of any action brought
//  by a third party) even if such damage or loss was
//  reasonably foreseeable or Xilinx had been advised of the
//  possibility of the same.
//
//  CRITICAL APPLICATIONS
//  Xilinx products are not designed or intended to be fail-
//  safe, or for use in any application requiring fail-safe
//  performance, such as life-support or safety devices or
//  systems, Class III medical devices, nuclear facilities,
//  applications related to the deployment of airbags, or any
//  other applications that could lead to death, personal
//  injury, or severe property or environmental damage
//  (individually and collectively, "Critical
//  Applications"). Customer assumes the sole risk and
//  liability of any use of Xilinx products in Critical
//  Applications, subject only to applicable laws and
//  regulations governing limitations on product liability.
//
//  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
//  PART OF THIS FILE AT ALL TIMES. 
//-----------------------------------------------------------------------------
//
// AXI Slave Terminator
//   Dummy AXI Slave core that sets all outputs to GND 
//
// Verilog-standard:  Verilog 2001
//--------------------------------------------------------------------------
//
// Structure:
//   axi_slave_term
//
//--------------------------------------------------------------------------

module axi_slave_term
(
s_axi_aclk,
s_axi_aresetn,
s_axi_awid,
s_axi_awaddr,
s_axi_awlen,
s_axi_awsize,
s_axi_awburst,
s_axi_awlock,
s_axi_awprot,
s_axi_awqos,
s_axi_awcache,
s_axi_awuser,
s_axi_awvalid,
s_axi_awready,

s_axi_wid,
s_axi_wdata,
s_axi_wstrb,
s_axi_wlast,
s_axi_wuser,
s_axi_wvalid,
s_axi_wready,

s_axi_bid,
s_axi_bresp,
s_axi_buser,
s_axi_bvalid,
s_axi_bready,

s_axi_arid,
s_axi_araddr,
s_axi_arlen,
s_axi_arsize,
s_axi_arburst,
s_axi_arlock,
s_axi_arprot,
s_axi_arqos,
s_axi_arcache,
s_axi_aruser,
s_axi_arvalid,
s_axi_arready,

s_axi_rid,
s_axi_rdata,
s_axi_rresp,
s_axi_rlast,
s_axi_ruser,
s_axi_rvalid,
s_axi_rready
);

parameter C_AXI_PROTOCOL      = "AXI4";
parameter C_AXI_ID_WIDTH      = 1;
parameter C_AXI_ADDR_WIDTH    = 32;
parameter C_AXI_DATA_WIDTH    = 32;
parameter C_AXI_AWUSER_WIDTH  = 4;
parameter C_AXI_WUSER_WIDTH   = 4;
parameter C_AXI_BUSER_WIDTH   = 4;
parameter C_AXI_ARUSER_WIDTH  = 4;
parameter C_AXI_RUSER_WIDTH   = 4;

input                               s_axi_aclk;
input                               s_axi_aresetn;
input   [C_AXI_ID_WIDTH-1: 0]       s_axi_awid;
input   [C_AXI_ADDR_WIDTH-1: 0]     s_axi_awaddr;
input   [7: 0]                      s_axi_awlen;
input   [2: 0]                      s_axi_awsize;
input   [1: 0]                      s_axi_awburst;
input   [1: 0]                      s_axi_awlock;
input   [2: 0]                      s_axi_awprot;
input   [3: 0]                      s_axi_awqos;
input   [3: 0]                      s_axi_awcache;
input   [C_AXI_AWUSER_WIDTH-1: 0]   s_axi_awuser;
input                               s_axi_awvalid;
output                              s_axi_awready;

// Write Data Channel
input   [C_AXI_ID_WIDTH-1: 0]       s_axi_wid;
input   [C_AXI_DATA_WIDTH-1: 0]     s_axi_wdata;
input   [C_AXI_DATA_WIDTH/8-1: 0]   s_axi_wstrb;
input                               s_axi_wlast;
input   [C_AXI_WUSER_WIDTH-1: 0]    s_axi_wuser;
input                               s_axi_wvalid;
output                              s_axi_wready;

// Read Response Channel
output  [C_AXI_ID_WIDTH-1: 0]       s_axi_bid;
output  [1: 0]                      s_axi_bresp;
output  [C_AXI_BUSER_WIDTH-1: 0]    s_axi_buser;
output                              s_axi_bvalid;
input                               s_axi_bready;

// Read Address Channel
input   [C_AXI_ID_WIDTH-1: 0]       s_axi_arid;
input   [C_AXI_ADDR_WIDTH-1: 0]     s_axi_araddr;
input   [7: 0]                      s_axi_arlen;
input   [2: 0]                      s_axi_arsize;
input   [1: 0]                      s_axi_arburst;
input   [1: 0]                      s_axi_arlock;
input   [2: 0]                      s_axi_arprot;
input   [3: 0]                      s_axi_arqos;
input   [3: 0]                      s_axi_arcache;
input   [C_AXI_ARUSER_WIDTH-1: 0]   s_axi_aruser;
input                               s_axi_arvalid;
output                              s_axi_arready;

// Read Data Channel
output  [C_AXI_ID_WIDTH-1: 0]       s_axi_rid;
output  [C_AXI_DATA_WIDTH-1: 0]     s_axi_rdata;
output  [1: 0]                      s_axi_rresp;
output                              s_axi_rlast;
output  [C_AXI_RUSER_WIDTH-1: 0]    s_axi_ruser;
output                              s_axi_rvalid;
input                               s_axi_rready;

////////////////////////////////////////////////////////////////////////////////
// Logic
////////////////////////////////////////////////////////////////////////////////

assign s_axi_awready  = 1'b0;
assign s_axi_wready   = 1'b0;

assign s_axi_bid      = {C_AXI_ID_WIDTH{1'b0}};
assign s_axi_bresp    = 2'b0;
assign s_axi_buser    = {C_AXI_BUSER_WIDTH{1'b0}};
assign s_axi_bvalid   = 1'b0;
assign s_axi_arready  = 1'b0;
assign s_axi_rready   = 1'b0;

endmodule

// XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
