docker run --runtime=nvidia --gpus all -v /var/run/docker.sock:/var/run/docker.sock -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy $*
