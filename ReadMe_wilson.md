# docker run 

docker run --shm-size=16g -itd --gpus all --init --net=host -v /home/wilsxue/APA/FlashOcc-sim:/workspace/FlashOcc-sim -v /home/Cnworkspace/nuscenes:/workspace/FlashOcc-sim/data/nuscenes --name bevaxocc_wilson bevdepth_parking:v2 /bin/bash


docker run -e DISPLAY=${DISPLAY:-0} \
           -e QT_X11_NO_MITSHM=1 \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           -v ${XAUTHORITY:-$HOME/.Xauthority}:/tmp/.Xauthority \
           --shm-size=16g \
           -itd \
           --gpus all \
           --init \
           --net=host \
           -v /home/wilsxue/APA/FlashOcc-sim:/workspace/FlashOcc-sim \
           -v /home/Cnworkspace/nuscenes:/workspace/FlashOcc-sim/data/nuscenes \
           --name bevaxocc_wilson \
           bevaxocc:v1 \
           /bin/bash
