import pandas as pd
from tqdm import tqdm

emotions_10 = ['anger','anticipation','disgust','fear','joy','negative', 'positive', 'sadness', 'surprise', 'trust']
emotions_8 = ['anger','anticipation','disgust','fear','joy', 'sadness', 'surprise', 'trust']


def synthesize(input,day):
    print(type(input) is str)
    if type(input) is str:
        df = pd.read_csv(input)
    elif type(input) is pd.DataFrame:
        df = input
    
    if len(df.columns) == 10:
        emotions = emotions_8
    elif len(df.columns) == 12:
        emotions = emotions_10

    print(emotions)
    print("df",df)
    count = df['Category'].value_counts()
    diff = set(emotions)-set(count.index)
    count = count.append(pd.Series([0]*len(diff),diff)).to_frame().transpose()
    # print(count)
    total = count.sum(axis = 1).values[0]
    count = 100*count.div(total)
    count['Dia'] = day
    # result = pd.concat([df,count])
    print(count)
    return count

def categorize(input):
    if type(input) is str:
        df = pd.read_csv(input)
    elif type(input) is pd.DataFrame:
        df = input
    print(df)
    valores = df.iloc[:,1:]
    valores = valores.astype('int32')
    df['Category'] = valores.idxmax(axis = 1)
    # table['Category'] = table.loc[:,table.columns[1]:].idxmax(axis=1)

    df['Category'][valores.max(axis = 1) == 0] = "Nenhum"

    return df


def read_dictionary(file):
    print("Reading Dictionary File")
    dictionary = {}
    df = pd.DataFrame(columns=['word']+emotions_10)
    with open(file,'r') as fileIn:
        word = ""
        info = []
        i = 0
        for line in tqdm(fileIn):
            # print(line)
            # print(info)
            p = line.strip().split('\t')
            # if p[0] != word:
            if i == 10:
                df.loc[len(df)] = [word]+info
                word = p[0]
                info = [float(p[2])]
                dictionary[word] = info
                i = 0

            else:
                info.append(float(p[2]))
            i+=1

    print("Finished reading file")
    return df,dictionary