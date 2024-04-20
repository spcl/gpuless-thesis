# API remoting for 10 warms runs of rodinia benchmarks
# and 10 native run
# Therefore we will run 15 runs in total, the first 5 will be seen as cold runs.

# Make sure to have gpuless running
ip='127.0.0.1'
root='/users/pzhou/projects/newgpuless/gpuless'
benchmarks=$root/src/benchmarks

echo
echo 'local execution'
echo

echo 'bfs'
./benchmark-api-remoting-scientific.sh $root $benchmarks/bfs native bfs $ip local

echo 'gaussian'
./benchmark-api-remoting-scientific.sh $root $benchmarks/gaussian native pathfinder $ip local

echo 'hotspot'
./benchmark-api-remoting-scientific.sh $root $benchmarks/hotspot native hotspot $ip local

echo 'pathfinder'
./benchmark-api-remoting-scientific.sh $root $benchmarks/pathfinder native pathfinder $ip local

echo 'srad_v1'
./benchmark-api-remoting-scientific.sh $root $benchmarks/srad_v1 native srad $ip local

echo
echo 'local TCP execution'
echo

echo 'bfs'
./benchmark-api-remoting-scientific.sh $root $benchmarks/bfs remote bfs $ip localtcp

echo 'gaussian'
./benchmark-api-remoting-scientific.sh $root $benchmarks/gaussian remote pathfinder $ip localtcp

echo 'hotspot'
./benchmark-api-remoting-scientific.sh $root $benchmarks/hotspot remote hotspot $ip localtcp

echo 'pathfinder'
./benchmark-api-remoting-scientific.sh $root $benchmarks/pathfinder remote pathfinder $ip localtcp

echo 'srad_v1'
./benchmark-api-remoting-scientific.sh $root $benchmarks/srad_v1 remote srad $ip localtcp