# Original docker run with GUI 
# docker run -it --shm-size=8gb --env="DISPLAY"  -v $(pwd):/home/haotian/jitnet-pytorch --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" detectron2:v2

# Custom run with bash on sora
nvidia-docker run -it --shm-size 12G -v $(pwd):/home/haotian/detectron2 -w /home/haotian/detectron2 detectron2:v2

# Custom run with bash on charlotte
nvidia-docker run -it --shm-size 32G -v $(pwd):/raid/haotian/Projects/detectron2 -w /home/haotian/detectron2 detectron2:v2


# docker login

# docker tag detectron2:v2 docker.io/haotianz/detectron2:v2

# docker push haotianz/detectron2:v2
