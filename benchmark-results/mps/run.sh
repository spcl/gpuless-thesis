# failed run
# sh run-bench-results.sh test11 mps-stream  &
# sh run-bench-results-defect.sh test11 mps-stream-defect
t='test13'
for i in $(seq 1 10); do
    echo "start $i"
    sh run-bench-results.sh $t mps-stream $i client1 &
    sh run-bench-results.sh $t mps-stream $i client3 &
    sh run-bench-results-defect.sh $t mps-stream-defect-8 $i

    # 11 
    sh run-bench-results.sh $t mps-stream $i client1 &
    ./mps-stream 0 >> /dev/null &
    # ./mps-stream 0 >> /dev/null
done


# good run
# sh run-bench-results.sh test12 mps-stream 0 &
# ./mps-stream 0