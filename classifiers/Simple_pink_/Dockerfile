FROM python:3.6
# FROM tensorflow/tensorflow


RUN pip3 install --upgrade pip
# RUN pip3 install tensorflow
# RUN echo pip -v
# RUN pip3 install torch torchvision
RUN pip3 install numpy
RUN pip3 install tensorflow 
RUN pip3 install pyyaml
RUN pip3 install keras --no-deps 
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install matplotlib


# RUN pip3 install nltk

#COPY commands.sh /opt/source-code/commands.sh
USER root
# RUN ["chmod", "+x", "opt/source-code/commands.sh"]


# ENTRYPOINT /bin/bash opt/source-code/commands.sh teste-gcn
#ENTRYPOINT /bin/bash opt/source-code/Simple/commands.sh
ENTRYPOINT ["opt/source-code/Simple_pink_/commands.sh"]

# COPY teste1/.vector_cache /opt/source-code/teste1/.vector_cache
# COPY teste1/*.py /opt/source-code/teste1/
# COPY Datasets /opt/source-code/Datasets
# COPY Embeddings /opt/source-code/Embeddings
# COPY teste1/variables.json /opt/source-code/teste1/


##ADD requirements.txt

#RUN pip3 install -r opt/source-code/requirements.txt

# RUN ls
# RUN ls opt/
# RUN ls opt/source-code/




