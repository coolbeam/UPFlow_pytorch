## How to Install

1. install the python environment
2. install the cuda correlation layer

## Python Environment

python3.5 is needed (training memory cost may be higher for python3.6 or higher in my case):
```
conda create -n upflow python=3.5
source deactivate
source activate upflow
```


use `pip install -r requirements.txt` to install python environment

 - Q1: ImportError: cannot import name 'DataLoaderIter

 - A1: DataLoaderIter is not exits in pytorch(1.2.0), may use _MultiProcessingDataLoaderIter or _SingleProcessDataLoaderIter: `from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as _DataLoaderIter`


## Cuda Correlation Layer
You should first check where your cuda is installed 

 - my case: python3.5 with cuda9.0, where the cuda in installed in /usr/local/cuda-9.0
 - another case: python3.5 with cuda10.0 installed in /data/cuda/cuda-10.0/cuda

Then check the 'cuda-path' in correlation_package/setup.py

install the correlation layer(maybe you should check your gcc version before compile the correlation layer, use `which gcc`):
```
cd ./model/correlation_package
python3 setup.py install --user
```

 - Q1: get error: OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
 - A1: use:
```
    export PATH=/data/cuda/cuda-10.0/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/data/cuda/cuda-10.0/cuda/lib64:/data/cuda/cuda-10.0/cudnn/v7.5.0/lib64:$LD_LIBRARY_PATH
```

 - Q2: permission denied
 - A2: try: `sudo python3 setup.py install` or `python3 setup.py install --user`


