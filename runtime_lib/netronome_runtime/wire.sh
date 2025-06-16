#!/bin/bash

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
# @file         apps/wire/init/wire.sh
# @brief        init script
#

NFP_SDK_DIR=${NFP_SDK_DIR:-/opt/netronome}
export PATH=${NFP_SDK_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${NFP_SDK_DIR}/lib:$LD_LIBRARY_PATH

CONFIG_DIR=$(dirname $0)
#output=/dev/null

OUTPUT="1"

Usage() {
        echo
        echo -e "\t ****** Error: $1 ****** "
        echo "Usage: $0 < start <mefw> | stop | restart <mefw> >"
        echo -e "\tstart   : Load ME fw and init MAC/TM/DMA"
        echo -e "\tstop    : Unload ME fw and stop MAC interfaces"
        echo -e "\trestart : stop & start"
        echo
        exit 1
}

case "$1" in
    start)
        echo "Starting FW:"

        echo -n " - Init MAC..."
        nfp-nsp -M &> $OUTPUT || exit 1
        echo "done"

        echo -n " - nfp-nsp Reset..."
        nfp-nsp -R &> $OUTPUT || exit 1
        echo "done"

        echo -n " - Load FW..."
        nfp-nffw $2 load --no-start &> $OUTPUT || exit 1
        echo "done"


        echo -n " - Init DMA..."
        nfp init dma 0 ${CONFIG_DIR}/nfp_nbi8_dma_i32_i33.json &> $OUTPUT || exit 1
        echo "done"


        echo -n " - Enable RX..."
        #nfp -m mac -e set port rx 0 0 enable &> $OUTPUT || exit 1
        #nfp -m mac -e set port rx 0 4 enable &> $OUTPUT || exit 1
        nfp -m mac -e set port rx 0 0 enable
        nfp -m mac -e set port rx 0 4 enable
        echo "done"

        echo -n " - Set EgressPrependEnable..."
        nfp-reg xpb:Nbi0IsldXpbMap.NbiTopXpbMap.MacGlbAdrMap.MacCsr.EgCmdPrependEn0Lo.EgCmdPrependEn=0xf000f
        echo "done"

        sleep 1

        echo -n " - Start ME's..."
        nfp-nffw start || exit 1
        echo "done"

        echo ""
        ;;
    reload|restart)
        if [ $# -lt 2 ]
        then
            Usage "Insufficient Parameters"
        fi
        $0 stop
        sleep 2
        $0 start $2
        ;;
    stop)

        echo "Stopping FW:"
        (rmmod nfp || true) 2> /dev/null
        modprobe nfp nfp_dev_cpp=1 nfp_pf_netdev=0
        echo -n " - Unload FW... "
        nfp-nffw unload
        echo "done"
        echo ""
        ;;
        *)
            Usage "Invalid option"
esac
exit 0
