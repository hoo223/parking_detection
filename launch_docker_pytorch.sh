sudo docker run --gpus all -it --net=host --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    -v `pwd`/share:/root/share \
    --name parking_torch \
    nvidia/cudagl:11.4.0-devel-ubuntu18.04 \
    bash
