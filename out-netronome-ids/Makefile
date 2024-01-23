#
# Copyright (C) 2015-2018,  Netronome Systems, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @file         apps/wire/Makefile
# @brief        Makefile for the ME cut-through wire app
#

# Define src_dir FIRST - it is the directory that this makefile resides in
# MUST OCCUR BEFORE ANY include's (which will change MAKEFILE_LIST)
app_src_dir     := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT_SRC_DIR  ?= $(realpath $(app_src_dir)/../..)
base_dir        := $(ROOT_SRC_DIR)

nfp6000_inc		:= $(base_dir)/include
me_blocks_dir   := $(base_dir)/microc/blocks
me_libs_dir     := $(base_dir)/microc/lib
me_inc_dir      := $(base_dir)/microc/include
scripts_dir     := $(base_dir)/scripts
ep2_dir      := $(base_dir)/lib

include $(scripts_dir)/Makefile.nfp.config
-include Makefile.nfp.config

all: core.fw

#
# Flags and Options
#
# blm_custom. This should be removed once BLM and NFD have default config.
# Common NFAS flags
mlir_NFASFLAGS += $(NFASFLAGS)
mlir_NFASFLAGS +=  $(mlir_APPDEFS) -DBLM_CUSTOM_CONFIG
mlir_NFASFLAGS += -DNFP_LIB_ANY_NFAS_VERSION
mlir_NFASFLAGS += \
	-I. \
	-I$(nfp6000_inc) \
	-I$(me_blocks_dir) \
	-I$(me_inc_dir) \
	-I$(me_libs_dir) \
	-I$(NFP_STD_LIB)/include \
	-I$(NFP_STD_LIB)/microcode/include \
	-I$(NFP_STD_LIB)/microcode/src \
	-I$(ep2_dir)

# Common NFCC flags
mlir_NFCCFLAGS += $(NFCCFLAGS)
mlir_NFCCFLAGS += -Qnctx_mode=8 -Qnctx=8
# mlir_NFCCFLAGS += -FI config.h
mlir_NFCCFLAGS += $(mlir_APPDEFS) -DBLM_CUSTOM_CONFIG -DIF_SIMULATION=1
mlir_NFCCFLAGS += \
	-I. \
	-I$(nfp6000_inc) \
	-I$(app_src_dir) \
	-I$(me_inc_dir) \
	-I$(me_libs_dir) \
	-I$(me_blocks_dir)/blm \
	-I$(me_blocks_dir)/blm/_h \
	-I$(NFP_STD_LIB)/include \
	-I$(NFP_STD_LIB)/microcode/include \
	-I$(NFP_STD_LIB)/microc/include \
	-I$(ep2_dir)

# Additional MicroC source files (libraries)
mlir_NFCCSRCS := \
	$(me_libs_dir)/nfp/libnfp.c \
	$(me_libs_dir)/pkt/libpkt.c \
	$(me_libs_dir)/std/libstd.c \
	$(me_libs_dir)/net/libnet.c \
	$(me_blocks_dir)/blm/libblm.c \
	$(NFP_STD_LIB)/microc/src/rtl.c \
	$(NFP_STD_LIB)/microc/src/intrinsic.c

# Common NFLD flags
mlir_NFLDFLAGS += -rtsyms -mip -chip $(CHIP)

global_LIST_FILES :=

#
# Infrastructure blocks
#

# BLM
BLM_DEFS := -DBLM_CUSTOM_CONFIG -DSINGLE_NBI -DPKT_NBI_OFFSET=$(PKT_NBI_OFFSET)
BLM_DEFS += -DBLM_BLQ_EMEM_TYPE=emem -DNBII=8 -DBLM_INSTANCE_ID=0
BLM_DEFS += -DBLM_INIT_EMU_RINGS

ME_BLM_SRC  := $(me_blocks_dir)/blm/blm_main.uc
ME_BLM_LIST := blm.list
ME_BLM_DEFS := $(mlir_NFASFLAGS) $(BLM_DEFS) -I. \
	-I$(app_src_dir) \
	-I$(me_blocks_dir)/blm/ \
	-I$(me_blocks_dir)/blm/_h \
	-I$(me_blocks_dir)/blm/_uc
$(ME_BLM_LIST): $(ME_BLM_SRC)
	@echo "--- Building $@"
	$(Q) $(NFAS) $(ME_BLM_DEFS) -o $@ $<
global_LIST_FILES += $(ME_BLM_LIST)


#
# Application
#
NET_SRCS := $(ep2_dir)/extern/extern_net.c
NET_LIST := dma.list
NET_DEFS := $(mlir_NFCCFLAGS)
$(NET_LIST): $(mlir_NFCCSRCS) $(NET_SRCS)
	@echo "--- Building $@"
	$(Q) $(NFCC) $(NET_DEFS) -Fe$@ $(NET_SRCS) $(mlir_NFCCSRCS)
global_LIST_FILES += $(NET_LIST)

DMA_SRCS := $(ep2_dir)/extern/extern_dma.c
DMA_LIST := dma.list
DMA_DEFS := $(mlir_NFCCFLAGS)
$(DMA_LIST): $(mlir_NFCCSRCS) $(DMA_SRCS)
	@echo "--- Building $@"
	$(Q) $(NFCC) $(DMA_DEFS) -Fe$@ $(DMA_SRCS) $(mlir_NFCCSRCS)
global_LIST_FILES += $(DMA_LIST)


S1_SRCS := $(app_src_dir)/msg_reassemble_1.c
S1_LIST := msg_reassemble_1.list
S1_DEFS := $(mlir_NFCCFLAGS)
$(S1_LIST): $(mlir_NFCCSRCS) $(S1_SRCS)
	@echo "--- Building #@"
	$(Q) $(NFCC) $(S1_DEFS) -Fe$@ $(S1_SRCS) $(mlir_NFCCSRCS)
global_LIST_FILES += $(S1_LIST)
S2_SRCS := $(app_src_dir)/process_packet_1.c
S2_LIST := process_packet_1.list
S2_DEFS := $(mlir_NFCCFLAGS)
$(S2_LIST): $(mlir_NFCCSRCS) $(S2_SRCS)
	@echo "--- Building #@"
	$(Q) $(NFCC) $(S2_DEFS) -Fe$@ $(S2_SRCS) $(mlir_NFCCSRCS)
global_LIST_FILES += $(S2_LIST)
core.fw: $(global_LIST_FILES)
	@echo "--- Linking $@"
	$(NFLD) $(mlir_NFLDFLAGS) \
	-elf $@ \
	-u mei0.me1 -l $(S2_LIST) \
	-u mei0.me2 -l $(S1_LIST) \
	-u mei0.me0 -l $(DMA_LIST) \
	-u ila0.me0 -l $(ME_BLM_LIST) \
	-i i8 -e $(PICO_CODE)



#
# Help
#
.PHONY : app_mlir_help
app_mlir_help:
	@echo "Build Options:"
	@echo "   Q                unset to print compiler output"
	@echo ""
	@echo "Path Settings:"
	@echo "   NFP_SDK_DIR      SDK installation directory"
	@echo ""
	@echo "Targets:"
	@echo "   help             this text"
	@echo "   clean            removes compiled binaries"
	@echo ""
	@echo "   wire.fw          wire application (default)"
	@echo "   mlir_dbg.fw      wire application with single app ME"
	@echo ""

all: core.fw

clean:
	rm -f *.list
	rm -f *.uci
	rm -f *.ucp
	rm -f *.obj
	rm -f *.fw

distclean: clean
