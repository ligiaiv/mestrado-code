from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
class myDataset(Dataset):

    def __init__(self,data,targets,tokenizer,max_len):
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,id):
        text = str(self.data[id])
        target = self.targets[id]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation= True,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
            )
        return {
            'text':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target,dtype = torch.long)

        }


def create_data_loader(df,tokenizer,max_len,batch_size):
    ds = myDataset(
        data = df.content.to_numpy(),
        targets = df.sentiment.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    print(len(ds[0]))
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )