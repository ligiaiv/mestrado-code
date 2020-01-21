from classifier import Classifier
from readFile import fileReader
import os, pandas, numpy,torch

#
#Read File
#
path = os.getcwd().split('/')
path[-1] = 'Datasets'
path = '/'.join(path)
reader  =  fileReader(path+"/tweets_hate_speech.csv",path+"/NAACL_SRW_2016.csv")
data,target = reader.readData()
print(data)

