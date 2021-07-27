# How to generate a personalized model

1. Capture your own dataset using [EyeCatcher](https://github.com/MSREnable/EyeCatcher)

2. Store this dataset on Azure Blob Storage

Note: Need to modify EyeCatcher to upload to Azure Blob Storage directly, along with some way to add upload auth tokens
See: https://github.com/Azure/azure-storage-net-data-movement

3. Spin up an Azure Data Science Hub (DSVM)

4. Copy the dataset down to a local /data directory using azcopy sync

azcopy_linux_amd64_10.11.0/azcopy sync "https://deepeyes.blob.core.windows.net/teamdrive?%token%" /data

5. git clone this repository onto the DSVM

6. Build a docker image from this repository, capturing all the ML requirements into the image

docker build -t gazepy .

7. Prep the data

dockRun.sh
    docker run --runtime=nvidia --gpus all -v /var/run/docker.sock:/var/run/docker.sock -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy $*

./dockRun.sh taskCreator.py --task 'prepareEyeCatcherTask' --input /data/200407 --output /data/out

./dockRun.sh taskCreator.py --task 'ROIDetectionTask' --input /data/out --output /data/out-meta --type rc

./dockRun.sh taskCreator.py --task 'ROIExtractionTask' --input /data/out --metapath /data/out-meta --output /data/out-prepped --type rc

8. Generate and test the model

./dockRun.sh main.py --data_path /data/out-prepped --reset
./dockRun.sh main.py --data_path /data/out-prepped --validate
./dockRun.sh main.py --data_path /data/out-prepped --test
