FROM python:3.6

RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install nltk


RUN python3 /opt/source-code/prepython.py

COPY commands.sh /opt/source-code/commands.sh
USER root

ENTRYPOINT /bin/bash opt/source-code/commands.sh

