FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM python:3.7

LABEL org.opencontainers.image.authors="MSREnable@Microsoft.Com"

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

RUN apt-get update && apt-get install -y build-essential cmake apt-utils
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

VOLUME [ "/app" ] # Dynamically mounted from the host
VOLUME [ "/data" ] # Dynamically mounted from the host

# COPY . /app # Now dynamically mounted from the host
WORKDIR /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8097

# Execute the python
ENTRYPOINT ["python"]