from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import time
import math
import numpy as np
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold 

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared = False)
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f'{str(asMinutes(s))} (remain {str(asMinutes(rs))})'
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{int(m)}m {int(s)}s'

class RMSELoss(nn.Module):
    def __init__(self, reduction = 'mean', eps = 1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'none')
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores



class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def prepare_input(vars,text):

    inputs = vars.tokenizer.encode_plus(
        text,
        return_tensors = None,
        add_special_tokens = True,
        max_length = vars.max_len,
        pad_to_max_length = True,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs   
def collate(inputs):
    mask_len = int(inputs['attention_mask'].sum(axis = 1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs

# def valid_prediction_debugger():



class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
    
        return attention_embeddings

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
     
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
       
        sum_mask = input_mask_expanded.sum(1)
       
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
      
        mean_embeddings = sum_embeddings/sum_mask
      
        return mean_embeddings

def round_off_point_5(number):
    return round(number * 2) / 2