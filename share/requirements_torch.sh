# For NVIDIA issue
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

sed -i 's/kr.archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list
apt update

apt install -y python3.8
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1


#pip3 install torch torchvision torchsummary
#apt install -y libgl1-mesa-glx libglib2.0-0
#pip3 install opencv-python jupyter matplotlib

