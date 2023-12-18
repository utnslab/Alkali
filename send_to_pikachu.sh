
set -e

ssh mihirs@192.168.0.8 "rm -f ~/ep2_backend_netronome/src/$1/*"
scp -r $1/ mihirs@192.168.0.8:~/ep2_backend_netronome/src/
ssh mihirs@192.168.0.8 "cp ~/ep2_backend_netronome/src/$1/* ~/ep2_backend_netronome/src/$2"
