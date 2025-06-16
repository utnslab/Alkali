cd src/$1
make clean
make all
cp ./core.fw ~/sim/examples/nfsim/packet_egress
