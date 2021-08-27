import pandas as pd
import nltk
from lemmatization import LemmatizerPT

#   FOR DEVELOPMENT
import random

meaning_list = []
word_dict = {}
quit()
with open("layout-one.txt",'r') as f_in:
    for i,line in enumerate(f_in):
        words = line.split('{')[1].split('}')[0]
        words = words.replace(' ','').split(',')
        meaning_list.append(words)
        for word in words:
            word_dict[word] = i

i = random.randint(0,len(word_dict))
word = list(word_dict.keys())[i]
print(word)
print("i: {}, word.keys[i]: {}, word[i]: {}, meaning_list[i]: {}".format(i,word,word_dict[word],meaning_list[word_dict[word]]))


DATA_FILE = "fortuna_labeled_data.csv"

df1 = pd.read_csv(DATA_FILE)
# print(df1.columns)
# a principio fazer pra todas as palavras da frase, depois trocar pra um numero reduzido de palabras
lemmatizer = LemmatizerPT()
print("lemmatizing sentences")
sentences = lemmatizer.lemmatization_pt(df1)
for row in df1:
    line = row['text']
    label = row['class']
    tokens = nltk.word_tokenize(line)
    #tokenizar
    #lematizar
    #for word in row:
        #checar condições
        #criar nova frase trocando por um sinonimo
        #juntar com " "
        #adicionar no df2 com a mesma label
    #juntar frase original no df2
df2 = pd.DataFrame(columns = df1.columns)
print(df2)