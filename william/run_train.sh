wget -O 'best.pt' 'https://www.dropbox.com/s/whstfav20633buu/best.pt?dl=1'
python3 train.py 's101_SGD_semi' \
                 --train-dir ~/data/DLCV/food_data/train \
                 --valid-dir ~/data/DLCV/food_data/val \
                 --seed 1116 \
                 --gpu-ids 0 \
                 --batch-size 32 \
                 --save-freq 2 \
                 --oversampling-thr 0.0000075 \
                 --num-workers 10 \
                 --num-epochs 30 \
                 --optimizer SGD \
                 --optimizer-settings '{"lr":1e-5, "weight_decay":5e-4, "momentum":0.9}' \
                 --lr-scheduler CustomScheduler \
                 --scheduler-settings '{"0": 1e-5, "30":0}'