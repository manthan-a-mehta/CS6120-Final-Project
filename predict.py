import torch
import pandas as pd
from data import ELLDataset
from config import Config
from torch.utils.data import DataLoader
from utils import collate,round_off_point_5,AverageMeter,RMSELoss
from model import DebertaModel,LSTM_regr,DistillBERTClass
import argparse


device = torch.device('cuda')
vars=Config("")
coh,sy,voc,phr,gr,con=[],[],[],[],[],[]
text_ids=[]

def op_to_submission(df,predictions):
    for i in range(len(predictions)):
        pred=predictions[i][0]
        # pred=[round_off_point_5(num) for num in pred]
        coh.append(pred[0])
        sy.append(pred[1])
        voc.append(pred[2])
        
        phr.append(pred[3])
        gr.append(pred[4])
        con.append(pred[5])
        text_ids.append(df.loc[i,"text_id"])
    rdf=pd.DataFrame({"text_id":text_ids,"cohesion":coh,"syntax":sy,
    "vocabulary":voc,"phraseology":phr,"grammar":gr,"conventions":con})
    return rdf


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--model_path',  type=str,
                    help='Path of the model to be used',required=True)
    parser.add_argument('--model',  type=str,
                    help='Model to be used(deb_fb3, LSTM, distill)',required=True)
    parser.add_argument('--test_path',  type=str,
                    help='path to test file',required=True)

    args = parser.parse_args()
    model_path=args.model_path
    model=args.model    
    df_test=pd.read_csv("data/train.csv").reset_index(drop = True)
    criterion=RMSELoss()
    losses=AverageMeter()
    vars.mod_pre=args.model
    if(vars.mod_pre=="deb_fb3"):
        print("here")
        model=DebertaModel()
    elif(vars.mod_pre=="LSTM"):
        model = LSTM_regr(35031,vars.lstm_emb_dim,vars.hidden_dim)
    elif(vars.mod_pre=="distill"):
        print("distilbert")
        model=DistillBERTClass()
    model.load_state_dict(torch.load(model_path)["model"])

    model.to(device)
    model.eval()    
    test_dataset = ELLDataset(vars,df_test,True)

    test_loader = DataLoader(test_dataset,
                              batch_size = 1,
                              shuffle = False, 
                              num_workers = vars.num_workers,
                              pin_memory = True, 
                              drop_last = False
                             )
    preds=[]
    for step, (inputs,labels) in enumerate(test_loader):
        attention_mask = inputs['attention_mask'].to(device)
        inputs = collate(inputs)
        # sys.exit()
        labels=labels.to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        preds.append(y_preds.detach().cpu().numpy())
        losses.update(loss.item(), 1)
    print("Average Loss",losses.avg)
    op=op_to_submission(df_test,preds)
    op.to_csv("submission.csv",index=False)

