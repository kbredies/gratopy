FROM python:3.10-bullseye
LABEL maintainer="Benjamin Hackl <benjamin.hackl@uni-graz.at>"

ENV TZ=Etc/UTC
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y \
        apt-utils \
	tar \
	curl \
	xz-utils \
	pocl-opencl-icd



RUN pip install -U pip

WORKDIR /usr/local/gratopy
ADD . .

RUN pip install -r requirements.txt
RUN pip install .

CMD ["/usr/local/bin/python"]
