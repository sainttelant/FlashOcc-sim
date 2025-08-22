# docker run 

docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size=16g -itd --gpus all --init --net=host -v /home/wilsxue/APA/FlashOcc-sim:/workspace/FlashOcc-sim -v /home/Cnworkspace/nuscenes:/workspace/FlashOcc-sim/data/nuscenes --name bevaxocc_wilson bevaxocc:v1 /bin/bash





