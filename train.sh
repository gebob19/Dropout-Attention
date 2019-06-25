
# IMDB
## Vanilla Dropout 
python train.py \
--IMDB \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--save-to=IMDB-VD-5K-june25-1200 
## Attention Dropout 
python train.py \
--IMDB \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--attention-dropout \
--save-to=IMDB-AD-5K-june25-1200 

# COLA
## Vanilla Dropout 
python train.py \
--COLA \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--save-to=COLA-VD-5K-june25-1200 
## Attention Dropout 
python train.py \
--COLA \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--attention-dropout \
--save-to=COLA-AD-5K-june25-1200

# QNLI
## Vanilla Dropout 
python train.py \
--QNLI \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--save-to=QNLI-VD-5K-june25-1200 
## Attention Dropout 
python train.py \
--QNLI \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--attention-dropout \
--save-to=QNLI-AD-5K-june25-1200

# QQP
## Vanilla Dropout 
python train.py \
--QQP \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--save-to=QQP-VD-5K-june25-1200 
## Attention Dropout 
python train.py \
--QQP \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--dset-size=5000 \
--save \
--attention-dropout \
--save-to=QQP-AD-5K-june25-1200

# RTE
## Vanilla Dropout 
python train.py \
--RTE \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--save \
--save-to=RTE-VD-5K-june25-1200 
## Attention Dropout 
python train.py \
--RTE \
--hidden-size=768 \
--n-heads=3 \
--n-layers=6 \
--batch-size=16 \
--max-epoch=2 \
--max-sent-len=512 \
--log-every=10 \
--validate-every=10 \
--n-valid=1000 \
--dropout=0.1 \
--lr=3e-5 \
--save \
--attention-dropout \
--save-to=RTE-AD-5K-june25-1200
