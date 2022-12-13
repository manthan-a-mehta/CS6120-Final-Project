import torch
from utils import AttentionPooling,MeanPooling
from torch.utils.data import Dataset
from config import Config
from transformers import AutoConfig,AutoModel
import torch.nn as nn
device = torch.device('cuda')
# from main import prepare_input
import sys
from transformers import DistilBertModel, DistilBertTokenizer
## Util Functions

class DebertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained("roberta-base", ouput_hidden_states = True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0
        
        
        #Backbone
        self.model = AutoModel.from_pretrained("roberta-base", config=self.config)
        
            
         
        # Last Fully Connected
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)
        # Average Pooling Layer
        self.pool = MeanPooling()

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            
            module.weight.data = nn.init.xavier_uniform_(module.weight.data)
           
                
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            
            
            module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            
                
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_() 
            module.weight.data.fill_(1.0)
    
    def feature(self, inputs):
        outputs = self.model(**inputs)
  

        last_hidden_states = outputs[0]
        # sys.exit()
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
      
            
        return feature
    
    def forward(self, inputs):
     
        feature = self.feature(inputs)        
        outout = self.fc(feature)
     
        return outout

class LSTM_regr(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)
        self.dropout = nn.Dropout(0.4).to(device)
        self.fc=nn.Linear(hidden_dim,6)
        
    def forward(self, x):
        x=x["input_ids"]
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.fc(ht[-1])

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("distilbert-base-cased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 6)
        self.pool = AttentionPooling(768)

    def forward(self, inputs):
        output_1 = self.l1(**inputs)
        last_hidden_states = output_1[0]
        pooler = self.pool(last_hidden_states, inputs['attention_mask'])
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class MetaModel(torch.nn.Module):
    
    def __init__(self):
        super(MetaModel, self).__init__()
        self.Model1=DistillBERTClass()
        self.Model2=DebertaModel()
        self.Model3=ElectraModel()
        self.weights=nn.Parameter(torch.tensor([0.333,0.333,0.334]))
    
    def forward(self,inputs):
        # Distillbert is the most powerfull
        output1=self.Model1(inputs)
        output2=self.Model2(inputs)
        output3=self.Model3(inputs)
        weights_normalized=nn.Softmax()(self.weights)
        # print(weights_normalized)
        return output1*weights_normalized[0]+output2*weights_normalized[1]+output3*weights_normalized[2]
    
class ElectraModel(torch.nn.Module):
    def __init__(self):
        super(ElectraModel, self).__init__()
        self.l1 = AutoModel.from_pretrained("cross-encoder/ms-marco-electra-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 6)
        self.pool = AttentionPooling(768)

    def forward(self, inputs):
        output_1 = self.l1(**inputs)
        last_hidden_states = output_1[0]
        pooler = self.pool(last_hidden_states, inputs['attention_mask'])
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

    