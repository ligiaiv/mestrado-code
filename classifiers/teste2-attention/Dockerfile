# FROM python:3
FROM tensorflow/tensorflow:latest-py3
RUN pip3 install keras
RUN pip3 install ktrain
RUN pip3 install pydot
RUN pip3 install graphviz
# COPY Datasets /opt/source-code/Datasets
COPY commands.sh /opt/source-code/

ENTRYPOINT opt/source-code/commands.sh teste2-attention



