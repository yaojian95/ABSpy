FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y --fix-missing build-essential
RUN apt-get install -y --fix-missing git wget unzip vim 
RUN apt-get install -y --fix-missing python3 python3-pip python3-dev
RUN pip3 install jupyter numpy scipy astropy healpy matplotlib progressbar seaborn pytest nbval

RUN mkdir /home/lab
WORKDIR /home/lab

#ABSpy
RUN git clone https://github.com/gioacchinowang/ABSpy.git abspy
WORKDIR /home/lab/abspy
RUN pip3 install .
WORKDIR /home/lab
