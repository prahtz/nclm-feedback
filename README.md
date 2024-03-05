# nclm-feedback

## Installation

```
apt-get install espeak-ng

pip install -r requirements.txt

# k2
# find the right version in https://huggingface.co/csukuangfj/k2
pip install https://huggingface.co/csukuangfj/k2/resolve/main/cpu/k2-1.24.4.dev20231220+cpu.torch2.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
cd -
source ~/.zshrc

# valle
git clone https://github.com/prahtz/valle.git
cd valle
pip install -e .
git clone https://github.com/y-ren16/TiCodec.git
cd ..
```