#!/bash

# First, need to download and install miniconda.
# See this page: https://docs.conda.io/en/latest/miniconda.html
# conda config --set auto_activate_base False
# Then, need to init correctly.

# Then, need to create a new env.
conda create --name capstone python=3.7.9
conda activate capstone

# Cuda is 9.0, so install the correct pytorch version
conda install -y cudatoolkit=10.1
conda install -y -c anaconda cudnn
conda install -y -c nvidia nvcc_linux-64
conda install -y numpy
conda install -y scipy
conda install -y scikit-learn
conda install -y matplotlib
# conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

# Maybe install from an env file instead?
