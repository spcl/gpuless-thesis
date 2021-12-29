PYTHONPATH=$HOME/libtorch LD_LIBRARY_PATH=$HOME/libtorch/lib:$HOME/cudnn/cuda/lib64:$LD_LIBRARY_PATH python run-batched.py 2>/dev/null > $(pwd | xargs basename)-hot-$(date --iso-8601=seconds).out
