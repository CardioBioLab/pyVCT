# pyVCT
Virtual Cardiac Tissue Model – A Cellular Potts Model for cardiac monolayers that reproduces fibrotic patterns

## Build instructions
`sudo apt-get install gcc-9`
`sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20`
`gcc --version`
`check if gcc version ==9.x.x`
`if not do: sudo update-alternatives --config gcc`
`choose gcc-9 version`
`pip install cython==0.29.32 pyyaml numpy jupyterlab wandb scikit-image feret pandas seaborn numpngw`  
`git clone https://github.com/CardioBioLab/pyVCT.git`  
`cd VCT`  
`make`  
`cd ..`  
`python3 setup.py build_ext --inplace`    
## Usage
`run example.ipynb file`
