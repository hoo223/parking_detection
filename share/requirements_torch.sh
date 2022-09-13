# For NVIDIA issue
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

sed -i 's/kr.archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list
apt update

# Python 3.8
apt install -y curl git gedit 
apt install -y python3.8 
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# pip
apt install -y python3-distuils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py

# Pytorch
pip install torch torchvision torchsummary
pip install opencv-python jupyter matplotlib
#apt install -y libgl1-mesa-glx libglib2.0-0


