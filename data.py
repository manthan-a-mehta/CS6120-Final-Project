import torch
from utils import AttentionPooling
from torch.utils.data import Dataset
from config import Config
from transformers import AutoConfig,AutoModel
import torch.nn as nn
from utils import prepare_input
## Util Functions



class ELLDataset(Dataset):
    """
    Main Dataset class
    """
    def __init__(self,vars,df,is_train=True):
        """_summary_

        Args:
            vars (config): Config Variable
            df (_type_): DataFrame to be parsed
            is_train (bool, optional): If it is train or not

        Returns:
            None
        """
#         self.cfg = cfg
        self.is_train=is_train
        self.vars=vars
        self.texts = df['full_text'].values
        if(self.is_train):
            self.labels = df[vars.target_cols].values
        
    def __len__(self):
        """_summary_

        Returns:
            int:  Length of text(Length of dataset or the batch)
        """
        return len(self.texts)
    
    def __getitem__(self, item):
        """_summary_

        Args:
            item (idx): idx of the batch to be retrieved from the dataloader

        Returns:
            inputs(torch.tensor): Returns batch of input data and labels of size BxLx768 and Bx6
        """
        if(self.is_train):
            inputs = prepare_input(self.vars,self.texts[item])
            label = torch.tensor(self.labels[item], dtype = torch.float)
            return inputs, label
        else:
            inputs = prepare_input(self.vars,self.texts[item])
            return inputs


