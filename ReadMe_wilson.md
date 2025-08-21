# docker run 

docker run --shm-size=16g -itd --gpus all --init --net=host -v /home/wilsxue/APA/FlashOcc-sim:/workspace/FlashOcc-sim -v /home/Cnworkspace/nuscenes:/workspace/FlashOcc-sim/data/nuscenes --name bevaxocc_wilson bevdepth_parking:v2 /bin/bash
