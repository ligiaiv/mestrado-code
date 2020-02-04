FROM python:3

RUN pip3 install torch torchvision
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install nltk
RUN pip3 install scikit-learn
RUN pip3 install sklearn
RUN pip3 install -U spacy
RUN pip3 install -U spacy-lookups-data
RUN pip3 install torchtext
RUN pip3 install matplotlib

#installing spacy dicts

RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download pt_core_news_sm

COPY . /opt/source-code
##ADD requirements.txt

#RUN pip3 install -r opt/source-code/requirements.txt

RUN ls
RUN ls opt/
RUN ls opt/source-code/

RUN chmod +x /opt/source-code/commands.sh
# CMD [ "python", "./my_script.py" ]

ENTRYPOINT opt/source-code/commands.sh


