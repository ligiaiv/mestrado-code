import os,re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from NLPyPort.FullPipeline import *
import sys
import stopwords_PT
LANG = "port"
NLPort_options = {
			"tokenizer" : True,
			"pos_tagger" : True,
			"lemmatizer" : True,
			"entity_recognition" : True,
			"np_chunking" : True,
			"pre_load" : False,
			"string_or_array" : False
}

bad_POS = ['punc']
if LANG is 'port':
	SW = stopwords_PT.stopwords
elif LANG is 'eng':
	SW = stopwords.words('english')

class LemmatizerPT():
	def __init__(self,options = NLPort_options):
		self.options = options	



	def check_ok(self,word,pos):
		if word in SW:
			return False
		if pos in bad_POS:
			return False
		
		return True
			


	def lemmatization_pt(self,df):

		np.savetxt("temp_text", df['text'].values, fmt='%s')
		# input_file="input_sample.txt"


		text=new_full_pipe("temp_text",options=self.options)
		print()
		if(text!=0):
			lemas = text.lemas
			tokens = text.tokens
			pos = text.pos_tags
			# print("TEXT")
			# print(tokens)
			# print(lemas)

			sentence = ""
			sentences = []
			for i in range(len(tokens)):
				lema = lemas[i]
				if "EOS" not in tokens[i]:
					if self.check_ok(lema,pos[i]):
						sentence += " "
						sentence += lema

				else:
					sentences.append(sentence)
					sentence = ""
		return sentences

# for i in range(21,22):

#le arquivo e manda para o classificador
# text = pd.read_csv('dados/28_pt_limpo.csv',sep='|')
# N = 1000
# lines = []
# with open('lem_28jan.txt', 'w') as f:

# 	for i in range(0, len(text), N):
# 		print("I",i)
# 		lines = lemmatization_pt(text[i:i + N])
# 		# lines = lines+lines_temp
# 		for line in lines:
# 			f.write(line+"\n")
# middle = int(len(text)/2)
# lines1 = lemmatization_pt(text[:middle])
# print('lines1 done')
# lines2 = lemmatization_pt(text[middle:])
# lines = lines1+lines2


