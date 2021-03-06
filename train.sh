# VAR SETUP
## Model Arch.
hidden_size="384"
n_heads="6"
n_layers="6"
## Training Params
batch_size="64"
epochs="2"
id="JUNE27"
dset_size="10000"

# IMDB
## Vanilla Dropout 
python train.py \
--IMDB \
--hidden-size=$hidden_size \
--n-heads=12 \
--n-layers=12 \
--batch-size=8 \
--max-epoch=1 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=50 \
--n-valid=2000 \
--dropout=0.3 \
--lr=3e-5 \
--save \
--save-to=IMDB-VD-$id 

## Attention Dropout 
python train.py \
--IMDB \
--hidden-size=$hidden_size \
--n-heads=12 \
--n-layers=12 \
--batch-size=8 \
--max-epoch=1 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=50 \
--n-valid=2000 \
--dropout=0.3 \
--lr=3e-5 \
--save \
--attention-dropout \
--save-to=IMDB-AD-$id 

## Entire Embedding Dropout 
python train.py \
--IMDB \
--hidden-size=$hidden_size \
--n-heads=12 \
--n-layers=12 \
--batch-size=8 \
--max-epoch=1 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=50 \
--n-valid=2000 \
--dropout=0.3 \
--lr=3e-5 \
--save \
--entire-embedding-dropout \
--save-to=IMDB-ED-$id 

# # # COLA
# ## Vanilla Dropout 
# python train.py \
# --COLA \
# --hidden-size=768 \
# --n-heads=$n_heads \
# --n-layers=$n_layers \
# --batch-size=$batch_size \
# --max-epoch=5 \
# --max-sent-len=512 \
# --log-every=20 \
# --validate-every=50 \
# --n-valid=1000 \
# --dropout=0.5 \
# --lr=3e-5 \
# --save \
# --save-to=COLA-VD-$id 

# ## Attention Dropout 
# python train.py \
# --COLA \
# --hidden-size=768 \
# --n-heads=$n_heads \
# --n-layers=$n_layers \
# --batch-size=$batch_size \
# --max-epoch=5 \
# --max-sent-len=512 \
# --log-every=20 \
# --validate-every=50 \
# --n-valid=1000 \
# --dropout=0.5 \
# --lr=3e-5 \
# --save \
# --attention-dropout \
# --save-to=COLA-AD-$id

# QNLI
## Vanilla Dropout 
python train.py \
--QNLI \
--hidden-size=$hidden_size \
--n-heads=6 \
--n-layers=6 \
--batch-size=$batch_size \
--max-epoch=3 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=100 \
--n-valid=6000 \
--dset-size=40000 \
--dropout=0.5 \
--lr=3e-5 \
--save \
--save-to=QNLI-VD-$id 
# Attention Dropout 
python train.py \
--QNLI \
--hidden-size=$hidden_size \
--n-heads=6 \
--n-layers=6 \
--batch-size=$batch_size \
--max-epoch=3 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=100 \
--n-valid=6000 \
--dset-size=40000 \
--dropout=0.5 \
--lr=3e-5 \
--save \
--attention-dropout \
--save-to=QNLI-AD-$id


# # QQP
# ## Vanilla Dropout 
# python train.py \
# --QQP \
# --hidden-size=$hidden_size \
# --n-heads=12 \
# --n-layers=12 \
# --batch-size=$batch_size \
# --max-epoch=1 \
# --max-sent-len=512 \
# --log-every=10 \
# --validate-every=100 \
# --n-valid=6000 \
# --dropout=0.5 \
# --lr=3e-5 \
# --dset-size=30000 \
# --save \
# --save-to=QQP-VD-12LAYER-$id 
# ## Attention Dropout 
# python train.py \
# --QQP \
# --hidden-size=$hidden_size \
# --n-heads=12 \
# --n-layers=12 \
# --batch-size=$batch_size \
# --max-epoch=1 \
# --max-sent-len=512 \
# --log-every=10 \
# --validate-every=100 \
# --n-valid=6000 \
# --dropout=0.5 \
# --lr=3e-5 \
# --dset-size=30000 \
# --save \
# --attention-dropout \
# --save-to=QQP-AD-12LAYER-$id


# ## Vanilla Dropout 
# python train.py \
# --QQP \
# --hidden-size=$hidden_size \
# --n-heads=6 \
# --n-layers=6 \
# --batch-size=$batch_size \
# --max-epoch=1 \
# --max-sent-len=512 \
# --log-every=10 \
# --validate-every=100 \
# --n-valid=6000 \
# --dropout=0.5 \
# --lr=3e-5 \
# --dset-size=30000 \
# --save \
# --save-to=QQP-VD-6LAYER-$id 
# ## Attention Dropout 
# python train.py \
# --QQP \
# --hidden-size=$hidden_size \
# --n-heads=6 \
# --n-layers=6 \
# --batch-size=$batch_size \
# --max-epoch=1 \
# --max-sent-len=512 \
# --log-every=10 \
# --validate-every=100 \
# --n-valid=6000 \
# --dropout=0.5 \
# --lr=3e-5 \
# --dset-size=30000 \
# --save \
# --attention-dropout \
# --save-to=QQP-AD-6LAYER-$id