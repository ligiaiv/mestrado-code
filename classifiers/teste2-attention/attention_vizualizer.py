import numpy as np
import json, random
import pandas as pd
from nltk.corpus import stopwords

N1 = 3
N2 = 200
N3 = 50


def print_stats(coisa,name):
    print("----------------------------\n{}".format(name))
    print("shape:{}\n{}".format(coisa.shape, coisa))
with open("teste2-attention/word_result.json") as jsonfile:
    attention_file =json.load(jsonfile)
# print(attention_file.keys())


stop_words = stopwords.words('english')
my_stop = ["they're","i'm","http","u",'None']
stop_words = stop_words+my_stop
sentences = attention_file["1"]["sentences"]
scores = attention_file["1"]["scores"]
scores =  np.array(scores)
scores = (scores==np.max(scores,axis=1)[:,None])
attention_weights = np.array(attention_file["1"]["attention_weights"])

print_stats(attention_weights,"attention_weights")

# sexist_attention_weights = attention_weights[scores[:,2] == 1]
#indices of the N1 more important words in each sentence 
indices =  (-attention_weights).argsort(axis = 1)[:,:N1]

print_stats(indices,"indices")
# indices = (-sexist_attention_weights).argsort(axis = 1)[:,:N1]
important_words =  np.take_along_axis(np.array(sentences),indices,1)

print_stats(scores,"scores")

print_stats(important_words,"important_words")
sexist_word_idx = important_words[scores[:,1] == 1]
# print(np.array(sentences).shape,indices.shape)
# print("S",sexist_word_idx)




# print(sexist_attention_weights.shape)
# quit()

with open("teste2-attention/idx2word.json") as jsonfile:

    word_dict_file =json.load(jsonfile)

id2word = word_dict_file["idx2word"]
id2word = {int(k):v for k,v in id2word.items()}

# word_sentences = np.vectorize(id2word.get)(np.array(sentences))
sexist_word_idx =sexist_word_idx.flatten() 
frequency = np.bincount(sexist_word_idx)

# sexist_words = np.vectorize(id2word.get)(sexist_word_idx).flatten()
print_stats(sexist_word_idx,"sexist_word_idx")
sexist_word_ids,frequency = np.unique(sexist_word_idx, return_counts=True)
# print(frequency)
most_sexist_ids = sexist_word_ids[(-frequency).argsort()[:N2]]
print(frequency[(-frequency).argsort()[:N2]])
most_sexist = np.vectorize(id2word.get)(np.array(most_sexist_ids))
cleaned_sexist = [word for word in most_sexist if word not in stop_words][:N3]
# print_stats(frequency,"frequency")
# print_stats(frequency.argsort()[:N2],"frequency.argsort()[:N2]")
# most_sexist = sexist_word_idx[frequency.argsort()[:N2]]
# word_sentences = np.array([list(map(id2word.get, sentence)) for sentence in sentences])
# sexist_classified_sentences = word_sentences[scores[:,2] == 1]

# sexist_words = np.take_along_axis(sentences,indices,1)


print(cleaned_sexist)
quit()
# for sentence in sentences:
#     # print(sentence)
#     # print(id2word)
#     print(list(map(id2word.get, sentence)))
#     quit()
# for x in random.sample(range(len(sentences)),10):

#     print(scores[x])
#     data = pd.DataFrame([word_sentences[x],attention_weights[x]]).transpose()
#     print(data)

    
