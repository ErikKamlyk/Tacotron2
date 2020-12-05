docker run -it \
    --gpus all \
    --net=host \
    --ipc=host \
    --detach \
    -v /home/$USER/Tacotron2/:/home/$USER \
    tacotron2