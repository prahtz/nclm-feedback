# nclm-feedback

## Installation

```
apt-get install espeak-ng

pip install -r requirements.txt

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