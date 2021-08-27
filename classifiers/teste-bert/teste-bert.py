from transformers import AutoTokenizer, BertTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

import torch
from extra import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd

RANDOM_SEED = 42
MAX_LEN = 160
BATCH_SIZE = 16
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("../Datasets/examples/reviews.csv")

def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else:
    return 2
df['sentiment'] = df.score.apply(to_sentiment)
class_names = ['negative', 'neutral', 'positive']



# quit()
# print(df.head())

# print(df.shape)


# PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-large-portuguese-cased'


# model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
# tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

sample_txt = 'Quando foi a última vez que eu saí? Eu estou em casa faz 2 semanas.'
# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)


# encoding = tokenizer.encode_plus(
#   sample_txt,
#   max_length=32,
#   truncation= True,
#   add_special_tokens=True, # Add '[CLS]' and '[SEP]'
#   return_token_type_ids=False,
#   padding=True,
#   return_attention_mask=True,
#   return_tensors='pt',  # Return PyTorch tensors
# )

# print(encoding['input_ids'][0])


df_train, df_test = train_test_split(
    df,
    test_size = 0.1,
    random_state = RANDOM_SEED
)

df_val, df_test = train_test_split(
    df_test,
    test_size = 0.5,
    random_state = RANDOM_SEED
)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


print(train_data_loader.__dict__)
data = next(iter(train_data_loader))
quit()

# data = train_data_loader.dataset[0]
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# print(tokens)
# print(token_ids)