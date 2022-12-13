from transformers import AutoTokenizer,DistilBertTokenizer
import json
class Config():
    def __init__(self,df_train):
        self.unique_string="distil_cased"
        self.target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.cross_cv_seed=42
        self.max_len=512
        self.model="distilbert-base-cased"
        self.tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.dot_token=self.tokenizer .convert_tokens_to_ids(["."])
        self.train=df_train
        self.output_dir="/home/balaji/manthan/ELL/outputs/"
        self.hidden_dim=768
        self.loss_func="RMSE"
        self.batch_size=16
        self.num_workers=4
        self.gpu_id=0
        self.device=f'cuda:{self.gpu_id}'
        self.encoder_lr=2e-5
        self.decoder_lr=2e-5
        self.weight_decay=0.01
        self.eps = 1e-6
        self.betas = (0.9, 0.999)
        self.num_warmup_steps=0
        self.scheduler="linear"
        self.lr=0.005
        self.num_warmup_steps=0
        self.epochs=10
        self.folds=6
        self.save_all_models=False
        self.max_grad_norm=1000
        self.print_freq=50
        self.apex = True
        self.cv_seed=42
        self.test_path="/home/balaji/manthan/ELL/outputs/distilbert-base-uncased_20_distill_768_scheduling_distil_fold0_best.pth"
        #deb_fb3,LSTM,distill
        self.mod_pre="distill"
        self.csv_save_key="{}_{}_{}_{}".format(self.model,self.epochs,self.mod_pre,self.unique_string)
        self.lstm_emb_dim=200
        self.num_folds_train=1
        self.scheduler_step_size=10
        self.scheduler_gamma=0.5
        self.debug=False
        
       
if __name__=="__main__":
    Config("")