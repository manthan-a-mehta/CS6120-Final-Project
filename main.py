import torch.nn as nn
import torch
import gc; gc.enable()
from torch.utils.data import DataLoader,Dataset
from IPython.display import clear_output
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import math
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold 
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer,AutoConfig,AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import time
import sys
from config import Config
from utils import AverageMeter,AttentionPooling,timeSince,get_score,RMSELoss,prepare_input,collate
from data import ELLDataset
from model import DebertaModel,LSTM_regr,DistillBERTClass,MetaModel
import logging
from predict import op_to_submission

## Argument parsing
parser = argparse.ArgumentParser(description='Training File')

parser.add_argument('--train_file_path',  type=str,
                    help='Path of the model to be used',required=True)
parser.add_argument('--debug',  type=int,
                    help='If in debugging mode',required=True)
parser.add_argument('--epochs',  type=int,
                    help='Number of epochs',required=True)
parser.add_argument('--logging_file_name',  type=str,
                    help='Number of epochs',required=True)


args = parser.parse_args()

device = torch.device('cuda')

train = pd.read_csv(args.train_file_path)
vars=Config(train)
tokenizer = vars.tokenizer
vars.epochs=args.epochs
# vars.max_len = 1428

vars.debug=args.debug

if(vars.debug):
    print("Debugging, would not save the model")
if(not vars.debug):
        logging.basicConfig(filename=args.logging_file_name, level=logging.INFO)
logging.info(vars.__dict__)
def kFold(k=5):
    """_summary_

    Args:
        k (int, optional): Number of folds. Defaults to 5.
        Assigns fold according to stratification
    """
       
    Fold = MultilabelStratifiedKFold(n_splits = k, shuffle = True, random_state = vars.cross_cv_seed)
    for n, (train_index, val_index) in enumerate(Fold.split(vars.train,vars.train[vars.target_cols])):
        vars.train.loc[val_index, 'fold'] = int(n)
    vars.train['fold'] = vars.train['fold'].astype(int)
   
kFold(vars.folds)
# print(vars.train.fold.value_counts())
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """

    Args:
        fold (int): fold being used for validation
        train_loader (utils.data.DataLoader): DataLoader for training data
        model (nn.Module): Model for training
        criterion (nn.RMSELoss): Objective/Loss function to optimize on
        optimizer (torch.optim): Optimizer used for updating weights
        epoch (int): Epoch used for training
        scheduler (torch.optim.lr_scheduler): Scheduler used for decaying the learning rate
        device (torch.device): GPU/CPU device to perform calculations on

    Returns:
        losses.avg (Average loss): Dictionary of AverageMeter Class,
    """
    losses = AverageMeter()
    model.train()

    start = end = time.time()
    global_step = 0
  
    for step, (inputs, labels) in enumerate(train_loader):
        attention_mask = inputs['attention_mask'].to(device)
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        y_preds = model(inputs)
        loss = criterion(y_preds, labels)
        loss.backward()
        optimizer.step()
        

        losses.update(loss.item(), batch_size)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), vars.max_grad_norm)

        end = time.time()
        if step % vars.print_freq == 0 or step == (len(train_loader) - 1):
            logging.info('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f} '
                  'LR: {lr:.8f} '
                  .format(epoch + 1, step, len(train_loader), remain = timeSince(start, float(step + 1)/len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr=optimizer.param_groups[0]["lr"]
                         )
                 )
    return losses.avg



def valid_fn(valid_loader, model, criterion, device):
    """
    Iterates over the entire validation dataset and calculates the loss and prediction
    Args:
    
        valid_loader (utils.data.DataLoader): DataLoader for validation data
        model (nn.Module): Model that has been trained and has to be used for validation
        criterion (nn.RMSELoss): Objective/Loss function to optimize on
        device (torch.device): GPU/CPU device to perform calculations on

    Returns:
        losses.avg (Average loss): Dictionary of AverageMeter Class,
        preds: Prediction on the validation data.
    """
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        # print(inputs["input_ids"])
       
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % vars.print_freq == 0 or step == (len(valid_loader) - 1):
            logging.info('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))
                         )
                 )
    return losses.avg, np.concatenate(preds)

def test_fn(model):
    """
    Iterates over the entire test dataset and calculaets he predictions and converts it to a csv.
    Args:
        model (nn.Module): Model that has been trained and has to be used for Testing
    Returns:
        op: pd.DataFrame: A DataFrame with all six columns, predictions for all the six parameters
    """
    df_test=pd.read_csv("data/test.csv")
    model.eval()
    test_dataset = ELLDataset(vars,df_test,False)

    test_loader = DataLoader(test_dataset,
                                batch_size = vars.batch_size,
                                shuffle = True, 
                                num_workers = vars.num_workers,
                                pin_memory = True, 
                                drop_last = True
                                )
    preds=[]
    epochs=[]
    for step, (inputs) in enumerate(test_loader):
            attention_mask = inputs['attention_mask'].to(device)
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            
            y_preds = model(inputs)
            preds.append(y_preds.detach().cpu().numpy())
            epochs.append(preds)
    
    op=op_to_submission(df_test,preds)

    return op

def train_loop(folds, fold):
    """
    Runs the training and validation on each fold of the DataFrame
    
    Args:
        folds: pd.DataFrame: The main training data to train on

    Returns:
        best_train_loss (float): Best Training Loss
        best_val_loss (float) : Best Validation loss
        valid_folds (int): Validation fold to perform validation on
        pd.concat([df_epoch, df_scores],axis = 1) (pd.DataFrame):concatenated score over each epoch.
        df_tests (pd.DataFrame): Test Dataset to perform testing on.
    """
    # print(folds.info())
    train_folds = folds[folds['fold'] != fold].reset_index(drop = True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop = True)
    valid_folds.to_csv("data/valid.csv")
    
    valid_labels = valid_folds[vars.target_cols].values
    
    train_dataset = ELLDataset(vars,train_folds)
    valid_dataset = ELLDataset(vars,valid_folds)
    
    train_loader = DataLoader(train_dataset,
                              batch_size = vars.batch_size,
                              shuffle = True, 
                              num_workers = vars.num_workers,
                              pin_memory = True, 
                              drop_last = True
                             )
    valid_loader = DataLoader(valid_dataset,
                              batch_size = vars.batch_size * 2,
                              shuffle=False,
                              num_workers=vars.num_workers,
                              pin_memory=True, 
                              drop_last=False)
    logging.info((vars.tokenizer.vocab_size))
    

   
    if(vars.mod_pre=="deb_fb3"):
        logging.info("Training Bert based model")
        model = DebertaModel()
    elif(vars.mod_pre=="LSTM"):
        logging.info("Training LSTM model")
        model=LSTM_regr(vars.tokenizer.vocab_size,vars.lstm_emb_dim,vars.hidden_dim)
    elif(vars.mod_pre=="distill"):
        logging.info("Training Distill bert")
        model=DistillBERTClass()
    elif(vars.mod_pre=="Meta"):
        logging.info("Training Meta Model")
        model=MetaModel() 
    else:
        raise ValueError("Enter values form 'distill','LSTM, 'deb_fb3','Meta'" )   
   
   # Freezing weights
    # for param in list(model.parameters())[:-2]:
    #     param.requires_grad = False

    model.to(vars.device)
    optimizer = AdamW(model.parameters(),vars.lr)
    scheduler = StepLR(optimizer, step_size=vars.scheduler_step_size, gamma=vars.scheduler_gamma)

    
    
    if vars.loss_func == 'SmoothL1':
        criterion = nn.SmoothL1Loss(reduction='mean')
    elif vars.loss_func == 'RMSE':
        criterion = RMSELoss(reduction='mean')
    
    best_score = np.inf
    best_train_loss = np.inf
    best_val_loss = np.inf
    
    epoch_list = []
    epoch_avg_loss_list = []
    epoch_avg_val_loss_list = []
    epoch_score_list = []
    epoch_scores_list = []
    df_tests=pd.DataFrame()
    for epoch in range(vars.epochs):
        start_time = time.time()
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, vars.device)


        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, vars.device)
        scheduler.step()
     
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time
        
        epoch_list.append(epoch+1)
        epoch_avg_loss_list.append(avg_loss)
        epoch_avg_val_loss_list.append(avg_val_loss)
        epoch_score_list.append(score)
        epoch_scores_list.append(scores)
        
        if best_score > score:
       
            best_score = score
            best_train_loss = avg_loss
            best_val_loss = avg_val_loss
            if(not vars.debug):
                print("Saving Model")
                torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        vars.output_dir + f"{vars.csv_save_key.replace('/', '-')}_fold{fold}_best.pth")
        
            
        if vars.save_all_models:
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        vars.output_dir + f"{vars.csv_save_key.replace('_', '-')}_fold{fold}_epoch{epoch + 1}.pth")

    df_tests=test_fn(model)
    predictions = torch.load(vars.output_dir + f"{vars.csv_save_key.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location = torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in vars.target_cols]] = predictions
    
    df_epoch = pd.DataFrame({'epoch' : epoch_list,
                             'MCRMSE' : epoch_score_list,
                             'train_loss' : epoch_avg_loss_list, 
                             'val_loss' : epoch_avg_val_loss_list})
    df_scores = pd.DataFrame(epoch_scores_list)
    df_scores.columns = vars.target_cols
    

    torch.cuda.empty_cache()
    gc.collect()
    
    return best_train_loss, best_val_loss, valid_folds, pd.concat([df_epoch, df_scores],axis = 1),df_tests

def run_training():
    """
    Performs training and validation across folds. Main controller function.
    """
    train_losses=[]
    val_losses=[]
    folds=[]
    for fold in range(vars.num_folds_train):
#         if fold in [CFG.trn_fold]:
            logging.info("=======Starting fold {}=========".format(fold))
            best_train_loss, best_val_loss, _oof_df, df_epoch_scores,df_test = train_loop(vars.train, fold)
            logging.info("Fold {} best train loss {}, best val loss {}".format(fold,best_train_loss,best_val_loss))

            folds.append(fold)
            train_losses.append(best_train_loss)
            val_losses.append(best_val_loss)
            if(not vars.debug):
                logging.info("Storing results to csv")
                df_epoch_scores.to_csv("results/{}_{}.csv".format(vars.csv_save_key,fold))
    if(not vars.debug):
            logging.info("Storing final submissions")
            df_test.to_csv("submission.csv")


if __name__=="__main__":
    print("=============Starting Training==========")
    run_training()