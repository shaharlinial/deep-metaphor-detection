# fresh linux installation
#!/bin/bash

# fresh linux installation

sudo apt-update
sudo apt-get install libpq-dev git
sudo apt install python3-pip

cd $HOME
git clone https://github.com/shaharlinial/deep-metaphor-detection.git
cd deep-metaphor-detection
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt-get install unzip
unzip glove.840B.300d.zip
mkdir glove
mv glove.840B.300d.txt glove/glove840B300d.txt

# cuda support is at:
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=U#buntu&target_version=1804&target_type=debnetwork