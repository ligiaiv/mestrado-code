import pandas as pd
from random import sample

INFILE = "Datasets/fortuna_labeled_data"
TIMES = 5
def create_sentence(sentence,times):
    new_sentences=[]
    words = sentence.split()
    idx = sample(range(0, len(words)), min(times,len(words)))
    for i in idx:
        words_ = words.copy()
        del words_[i]
        new_sentences.append(" ".join(words_))
    return new_sentences

df = pd.read_csv(INFILE+".csv")
for index, row in df.iterrows():
    new = create_sentence(row["text"],TIMES)
    y = row["class"]
    for n in new:
        df = df.append({"text":n,"class":y},ignore_index=True)
        # print(df.size)
    
df.to_csv("{}_extended_{}.csv".format(INFILE,TIMES))