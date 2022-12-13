VERSION=v17
ID=2
LOGFILE=logs/exp_${VERSION}.log
CUDA_VISIBLE_DEVICES=${ID} python3 main.py --train_file_path "data/train.csv" --debug 1 --epochs 1 --logging_file_name "logs_new/debugging.logs" > "$LOGFILE" 2>&1 &
