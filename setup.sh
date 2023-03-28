#!/bin/bash
# USES Python 3.7
conda env create -f environment.yml
conda activate eai_v3

which python
which python3

sudo apt install -y cmake libeigen3-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5
cd g2opy
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) -DPYBIND11_PYTHON_VERSION=3.7 ..
make -j8
cd ..
python setup.py install
