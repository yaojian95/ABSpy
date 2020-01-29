FROM ubuntu:latest

# users may need python3-tk, please install it manually: apt-get install python3-tk

RUN apt-get update
RUN apt-get install -y --fix-missing build-essential
RUN apt-get install -y --fix-missing git wget unzip vim python3 python3-pip python3-dev
RUN pip3 install jupyter numpy scipy astropy healpy matplotlib progressbar seaborn nbval

RUN mkdir /home/lab
WORKDIR /home/lab

#ABSpy
RUN git clone https://github.com/gioacchinowang/ABSpy.git abspy
WORKDIR /home/lab/abspy
RUN python3 setup.py install
WORKDIR /home/lab

# CLEAN PACKAGES
RUN rm -f *.tar.gz
# SET PATH
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}
