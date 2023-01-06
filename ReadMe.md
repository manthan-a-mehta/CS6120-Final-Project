# Using the Codebase

1) Unzip the folder using the following command:

```
$ unzip CS6120-final-project.zip
```

2) cd to the newly unzipped folder.

```
$ cd CS6120-final-project
```

3) Create and activate a virtual environment. Make sure you have python3 installed.

```
$ python3 -m venv env
$ source env/bin/activate
```

4) Install all the requirements:
```
$ pip install -r requirements.txt
``` 

5) Create folders
   
```
$ mkdir logs_new
$ mkdir outputs
$ mkdir results
```
set your self.output_dir to your current outputs folder to ensure your models are saved.

6) Now we are ready to train model. Open **config.py** and edit any hyperparameter you want for the training to start. Make sure you select the correct self.mod_pre variable. The flexibility is to ensure modularity in code for more experiments. 
   
    Inside the train.sh file which is the main trin file, change the training file (data/train.csv), debug (1 or 0 to set if you are debugging or not), --log_file_path which is the file were it will store logs and --epochs to specify number of epochs. For example:

'''
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --train_file_path "data/train.csv" --debug 1 --epochs 1 --logging_file_name "logs_new/debugging.logs" 

'''

7) Generate predictions on the test file. Make sure you have the same tokenizer that you used during your training in the config.py file. Execute this command and you will have the predictions inside submission.csv

```
python3 predict.py --model_path /home/balaji/manthan/ELL/CS6120-final-project/outputs/distilbert-base-cased_10_distill_distil_cased_fold0_best.pth --model "distill" --test_path "data/test.csv"
```
