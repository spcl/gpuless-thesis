PYTHONPATH=$HOME/libtorch LD_LIBRARY_PATH=$HOME/libtorch/lib:$HOME/cudnn/cuda/lib64:$LD_LIBRARY_PATH python run.py 2>/dev/null | tail -n1
