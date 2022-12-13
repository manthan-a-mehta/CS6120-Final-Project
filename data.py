import torch
from utils import AttentionPooling
from torch.utils.data import Dataset
from config import Config
from transformers import AutoConfig,AutoModel
import torch.nn as nn
from utils import prepare_input
## Util Functions



class ELLDataset(Dataset):
    def __init__(self,vars,df,is_train=True):
#         self.cfg = cfg
        self.is_train=is_train
        self.vars=vars
        self.texts = df['full_text'].values
        if(self.is_train):
            self.labels = df[vars.target_cols].values
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        if(self.is_train):
            inputs = prepare_input(self.vars,self.texts[item])
            label = torch.tensor(self.labels[item], dtype = torch.float)
            return inputs, label
        else:
            inputs = prepare_input(self.vars,self.texts[item])
            return inputs


