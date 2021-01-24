FROM ubuntu:18.04

RUN apt -y update &&\
    apt -y install python3.6 python3-pip
RUN pip3 install --upgrade pip

ADD ./requirements.txt /
RUN pip3 install -r requirements.txt

ADD ./flask_app.py /
ADD ./best_nn_model.h5 /


CMD [ "python3", "-u", "./flask_app.py" ]
