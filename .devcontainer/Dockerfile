# use latest python alphine image.
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

COPY requirement.txt /tmp/requirements.txt

# install system dependencies.
RUN apt-get update && apt-get install -y \
  gcc libc-dev g++ graphviz git bash imagemagick inkscape fonts-open-sans curl fontconfig

# install fonts
RUN curl -O https://noto-website.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip \
  && mkdir -p /usr/share/fonts/NotoSansCJKjp \
  && unzip NotoSansCJKjp-hinted.zip -d /usr/share/fonts/NotoSansCJKjp/ \
  && rm NotoSansCJKjp-hinted.zip \
  && fc-cache -fv

# install python requirements.
RUN pip install -r /tmp/requirements.txt
# RUN pip install black graphviz jinja2 diagrams