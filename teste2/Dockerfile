# FROM python:3
FROM tensorflow/tensorflow:latest-py3
# FROM tensorflow/tensorflow
# FROM amd64/debian:buster

# CMD python3 -V
# CMD echo python3 -V

# CMD echo pip -v

# RUN pip3 install --upgrade pip
# RUN pip3 install tensorflow
# # RUN pip3 install torch torchvision
# # RUN pip3 install pandas
# # RUN pip3 install numpy
# # RUN pip3 install nltk
# # RUN pip3 install scikit-learn
# # RUN pip3 install sklearn
# # RUN pip3 install -U spacy
# # RUN pip3 install -U spacy-lookups-data
# # RUN pip3 install torchtext
# # RUN pip3 install matplotlib
# # RUN pip3 install transformers
RUN pip3 install keras
RUN pip3 install ktrain
# # #installing spacy dicts

# # RUN python3 -m spacy download en_core_web_sm
# # RUN python3 -m spacy download pt_core_news_sm

# # #COPY teste1/.vector_cache /opt/source-code/teste1/.vector_cache
# COPY teste2/*.py /opt/source-code/teste2/
COPY Datasets /opt/source-code/Datasets
COPY commands.sh /opt/source-code/
# # #COPY teste1/variables.json /opt/source-code/teste1/


# ##ADD requirements.txt

# #RUN pip3 install -r opt/source-code/requirements.txt

# # RUN ls
# # RUN ls opt/
# # RUN ls opt/source-code/

# RUN chmod +x /opt/source-code/commands.sh
# # CMD [ "python", "./my_script.py" ]
# RUN mkdir -p opt/source-code/teste2/results
# CMD pip -V
ENTRYPOINT opt/source-code/commands.sh teste2



