# For NVIDIA issue
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

sed -i 's/kr.archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list
apt update

# Python 3.8
apt install -y curl gedit 
apt install -y python3.8 
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# pip
apt install -y python3-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
rm -f get-pip.py

# Pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 # https://powerofsummary.tistory.com/189
pip install opencv-python jupyter matplotlib torchsummary
#apt install -y libgl1-mesa-glx libglib2.0-0


