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
// AXI Master Terminator
//   Dummy AXI Master core that sets all outputs to GND 
//
// Verilog-standard:  Verilog 2001
//--------------------------------------------------------------------------
//
// Structure:
//   axi_master_term
//
//--------------------------------------------------------------------------

module axi_master_term
(
m_axi_aclk,
m_axi_aresetn,
m_axi_awid,
m_axi_awaddr,
m_axi_awlen,
m_axi_awsize,
m_axi_awburst,
m_axi_awlock,
m_axi_awprot,
m_axi_awqos,
m_axi_awcache,
m_axi_awuser,
m_axi_awvalid,
m_axi_awready,

m_axi_wid,
m_axi_wdata,
m_axi_wstrb,
m_axi_wlast,
m_axi_wuser,
m_axi_wvalid,
m_axi_wready,

m_axi_bid,
m_axi_bresp,
m_axi_buser,
m_axi_bvalid,
m_axi_bready,

m_axi_arid,
m_axi_araddr,
m_axi_arlen,
m_axi_arsize,
m_axi_arburst,
m_axi_arlock,
m_axi_arprot,
m_axi_arqos,
m_axi_arcache,
m_axi_aruser,
m_axi_arvalid,
m_axi_arready,

m_axi_rid,
m_axi_rdata,
m_axi_rresp,
m_axi_rlast,
m_axi_ruser,
m_axi_rvalid,
m_axi_rready
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

input                               m_axi_aclk;
input                               m_axi_aresetn;
output  [C_AXI_ID_WIDTH-1: 0]       m_axi_awid;
output  [C_AXI_ADDR_WIDTH-1: 0]     m_axi_awaddr;
output  [7: 0]                      m_axi_awlen;
output  [2: 0]                      m_axi_awsize;
output  [1: 0]                      m_axi_awburst;
output  [1: 0]                      m_axi_awlock;
output  [2: 0]                      m_axi_awprot;
output  [3: 0]                      m_axi_awqos;
output  [3: 0]                      m_axi_awcache;
output  [C_AXI_AWUSER_WIDTH-1: 0]   m_axi_awuser;
output                              m_axi_awvalid;
input                               m_axi_awready;

// Write Data Channel
output  [C_AXI_ID_WIDTH-1: 0]       m_axi_wid;
output  [C_AXI_DATA_WIDTH-1: 0]     m_axi_wdata;
output  [C_AXI_DATA_WIDTH/8-1: 0]   m_axi_wstrb;
output                              m_axi_wlast;
output  [C_AXI_WUSER_WIDTH-1: 0]    m_axi_wuser;
output                              m_axi_wvalid;
input                               m_axi_wready;

// Read Response Channel
input   [C_AXI_ID_WIDTH-1: 0]       m_axi_bid;
input   [1: 0]                      m_axi_bresp;
input   [C_AXI_BUSER_WIDTH-1: 0]    m_axi_buser;
input                               m_axi_bvalid;
output                              m_axi_bready;

// Read Address Channel
output  [C_AXI_ID_WIDTH-1: 0]       m_axi_arid;
output  [C_AXI_ADDR_WIDTH-1: 0]     m_axi_araddr;
output  [7: 0]                      m_axi_arlen;
output  [2: 0]                      m_axi_arsize;
output  [1: 0]                      m_axi_arburst;
output  [1: 0]                      m_axi_arlock;
output  [2: 0]                      m_axi_arprot;
output  [3: 0]                      m_axi_arqos;
output  [3: 0]                      m_axi_arcache;
output  [C_AXI_ARUSER_WIDTH-1: 0]   m_axi_aruser;
output                              m_axi_arvalid;
input                               m_axi_arready;

// Read Data Channel
input   [C_AXI_ID_WIDTH-1: 0]       m_axi_rid;
input   [C_AXI_DATA_WIDTH-1: 0]     m_axi_rdata;
input   [1: 0]                      m_axi_rresp;
input                               m_axi_rlast;
input   [C_AXI_RUSER_WIDTH-1: 0]    m_axi_ruser;
input                               m_axi_rvalid;
output                              m_axi_rready;

////////////////////////////////////////////////////////////////////////////////
// Logic
////////////////////////////////////////////////////////////////////////////////

assign m_axi_awid     = {C_AXI_ID_WIDTH{1'b0}};
assign m_axi_awaddr   = {C_AXI_ADDR_WIDTH{1'b0}};
assign m_axi_awlen    = 8'b0;
assign m_axi_awsize   = 3'b0;
assign m_axi_awburst  = 2'b0;
assign m_axi_awlock   = 2'b0;
assign m_axi_awprot   = 3'b0;
assign m_axi_awqos    = 4'b0;
assign m_axi_awcache  = 4'b0;
assign m_axi_awuser   = {C_AXI_AWUSER_WIDTH{1'b0}};
assign m_axi_awvalid  = 1'b0;

assign m_axi_wid      = {C_AXI_ID_WIDTH{1'b0}};
assign m_axi_wdata    = {C_AXI_DATA_WIDTH{1'b0}};
assign m_axi_wstrb    = {C_AXI_DATA_WIDTH/8{1'b0}};
assign m_axi_wlast    = 1'b1;
assign m_axi_wuser    = {C_AXI_WUSER_WIDTH{1'b0}};
assign m_axi_wvalid   = 1'b0;

assign m_axi_bready   = 1'b0;

assign m_axi_arid     = {C_AXI_ID_WIDTH{1'b0}};
assign m_axi_araddr   = {C_AXI_ADDR_WIDTH{1'b0}};
assign m_axi_arlen    = 8'b0;
assign m_axi_arsize   = 3'b0;
assign m_axi_arburst  = 2'b0;
assign m_axi_arlock   = 2'b0;
assign m_axi_arprot   = 3'b0;
assign m_axi_arqos    = 4'b0;
assign m_axi_arcache  = 4'b0;
assign m_axi_aruser   = {C_AXI_ARUSER_WIDTH{1'b0}};
assign m_axi_arvalid  = 1'b0;

assign m_axi_rready   = 1'b0;

endmodule

// XSIP watermark, do not delete 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
