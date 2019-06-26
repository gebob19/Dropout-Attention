# VAR SETUP
## Model Arch.
hidden_size="384"
n_heads="12"
n_layers="12"
## Training Params
batch_size="32"
epochs="2"
id="JUNE25-10K-12LAYER"
dset_size="10000"

# RTE
## Vanilla Dropout 
python train.py \
--RTE \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--save \
--save-to=RTE-VD-$id 
## Attention Dropout 
python train.py \
--RTE \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--save \
--attention-dropout \
--save-to=RTE-AD-$id

# IMDB
## Vanilla Dropout 
python train.py \
--IMDB \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--save-to=IMDB-VD-$id 
## Attention Dropout 
python train.py \
--IMDB \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--attention-dropout \
--save-to=IMDB-AD-$id 

# COLA
## Vanilla Dropout 
python train.py \
--COLA \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--save-to=COLA-VD-$id 

## Attention Dropout 
python train.py \
--COLA \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--attention-dropout \
--save-to=COLA-AD-$id

# QNLI
## Vanilla Dropout 
python train.py \
--QNLI \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--save-to=QNLI-VD-$id 
## Attention Dropout 
python train.py \
--QNLI \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--attention-dropout \
--save-to=QNLI-AD-$id

# QQP
## Vanilla Dropout 
python train.py \
--QQP \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--save-to=QQP-VD-$id 
## Attention Dropout 
python train.py \
--QQP \
--hidden-size=$hidden_size \
--n-heads=$n_heads \
--n-layers=$n_layers \
--batch-size=$batch_size \
--max-epoch=$epochs \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=$dset_size \
--save \
--attention-dropout \
--save-to=QQP-AD-$id

