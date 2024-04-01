FROM ubuntu:20.04

RUN apt update && \
    apt-get install -y curl python3 python3-pip && \
    python3 -m pip install virtualenv

RUN mkdir -p /Projet_Linux_Image

WORKDIR /Projet_Linux_Image
COPY . .

RUN bash install.sh

CMD ["bash", "launch.sh"]