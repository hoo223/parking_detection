sudo docker run --gpus all -it --net=host --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    -v `pwd`/share:/root/share \
    --name parking_labeling \
    python:3.8.13-slim-buster \
    bash
