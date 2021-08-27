# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import re,json
from tqdm import tqdm

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from emolex_aux import *

LANG = "eng"
FILEIN = 'dados/28_en_limpo.csv'
if LANG == "port":
    DICT_FILE = 'dict_PT.json'
elif LANG == "eng":
    DICT_FILE = 'dict.json'


wordnet_lemmatizer = WordNetLemmatizer()
SW = stopwords.words('english')



# df,dictionary = read_dictionary('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
# df,dictionary = read_dictionary('emolexDictionaryPort.txt')

# with open(DICT_FILE, 'w') as json_file:
#     json.dump(dictionary, json_file,ensure_ascii=False)
# quit()
with open(DICT_FILE, 'r') as json_file:
    dictionary = json.load(json_file)

def classify(lines,name):
    pd_out = pd.DataFrame(columns=['word']+emotions_10)
    print("pd_out",pd_out)
    j = 0
    print(j)
    for line in tqdm(lines):
        # if line is None:
        #     continue
        try:
            line = line.strip("\n")
        except:
            print("deu ruim")
            continue
        words = re.split(";|,|\ |\*|\n|\.|\(|\)|!|W|:",line)
        if LANG is 'eng':
            words = list(map(wordnet_lemmatizer.lemmatize,words))
            words = list(filter(lambda a: a not in SW, words))

        score = np.zeros(10)
        for word in words:
            # print(word)
            values = dictionary.get(word)
            # print(values)
            if values is not None:
                score+=np.array(values)
                # print('here2')
        
        pd_out.loc[len(pd_out)] = [line]+score.tolist()

        if len(pd_out)>1000:
            pd_out_8 = pd_out.copy()
            pd_out_8 = pd_out_8.drop(columns=['positive', 'negative'])

            pd_out_10 = categorize(pd_out.copy())
            pd_out_8 = categorize(pd_out_8)


            pd_out_10.to_csv(name+"_categorized_tweets_10_"+LANG+".csv", mode='a', header=False, index = False)
            pd_out_8.to_csv(name+"_categorized_tweets_8_"+LANG+".csv", mode='a', header=False, index = False)

            pd_out = pd.DataFrame(columns=['word']+emotions_10)

        # print("pdout",pd_out)
        # print('here')
    # print("len_out",len(pd_out))



    # pd_out_8 = pd_out.copy()
    # pd_out_8 = pd_out_8.drop(columns=['positive', 'negative'])

    # # print(pd_out.copy())


    # pd_out_10 = categorize(pd_out.copy())
    # pd_out_8 = categorize(pd_out_8)


    # pd_out_10.to_csv(name+LANG+"_out_emolex_.csv", mode='a', header=False)

    # pd_out_8.to_csv(name+LANG+"_out_emolex_8_pt.csv", mode='a', header=False)
    # print("Antes",pd_out_10)
    return pd_out_10,pd_out_8

pd_total_10 = pd.DataFrame(columns=['Dia']+emotions_10)
pd_total_8 = pd.DataFrame(columns=['Dia']+emotions_8)

# for i in range(2):
names = ['pt','en']
i = 28
# print("Processando dia ",i)
#le arquivo e manda para o classificador
# text = pd.read_csv('dados/'+str(i)+'jan/top_tweets.csv')["text"].tolist()
if LANG is "port":
    text = pd.read_csv(FILEIN,header=None,sep='\t').loc[:,0].tolist()
# print(text)
# quit()
elif LANG is "eng":
    text = pd.read_csv(FILEIN,sep='|')['text'].tolist()


pd_out_10,pd_out_8 = classify(text,str(i))

# sintetiza a informacao do dia
# count = pd_out_10['Category'].value_counts()
# diff = set(emotions_10)-set(count.index)
# count = count.append(pd.Series([0]*len(diff),diff)).to_frame().transpose()
# print("SUM",count.sum(axis = 1).values[0])
# print(count)
# total = count.sum(axis = 1).values[0]
# count = 100*count.div(total)
# print(count)
# count['Dia'] = i
# pd_total_10 = pd.concat([pd_total_10,count])

# count = pd_out_8['Category'].value_counts()
# diff = set(emotions_8)-set(count.index)
# count = count.append(pd.Series([0]*len(diff),diff)).to_frame().transpose()
# total = count.sum(axis = 1).values[0]

# count = 100*count.div(total)

# count['Dia'] = i
# pd_total_8 = pd.concat([pd_total_8,count])


# print(pd_total_8)

# pd_total_10.to_csv("28jan_10_en.csv")
# pd_total_8.to_csv("28jan_8_en.csv")


