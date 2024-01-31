
set -e
rm ./fpga_out/*
cp *sv ./fpga_out/
scp -r -oProxyCommand="ssh -W %h:%p jxlin@dex.csres.utexas.edu" ./fpga_out ditto:~/ep2_fpga_lib/
