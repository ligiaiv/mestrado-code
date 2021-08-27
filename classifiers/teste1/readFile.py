#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#   Script created to read dataset file
#

import pandas as pd
import os, json

class fileReader:
    def __init__(self,filename,targetname):
        self.filename = filename
        self.targetname = targetname

    def readData(self):
        self.readTarget()

        data = pd.read_csv(self.filename,delimiter="|", encoding='utf-8')
        data["class"] = data["id"].map(self.target)
        data = data[['id','text','class']]

        # data.to_csv("labeled_data.csv")
        # print(type(data))
        text = data['text'].tolist()
        label = data['class'].tolist()

        return(text,label)
    def readTarget(self):
        
        self.target = pd.read_csv(self.targetname,delimiter=",", encoding='utf-8').set_index('id').to_dict()['class']

        

# path = os.getcwd().split('/')
# path[-1] = 'Datasets'
# path = '/'.join(path)
# reader  =  fileReader(path+"/tweets_hate_speech.csv",path+"/NAACL_SRW_2016.csv")
# reader.readData()
