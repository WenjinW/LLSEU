# The command to start the experiment
Taking the PMNIST dataset as an example, we give the command tp start
experiments with different methods.
## run SGD on PMNIST
```
python run.py --experiment pmnist
              --approach ewc
              --model alexnet
              --epochs 50
              --batch 128
              --lr 0.025
              --lamb 0.0003
              --lamb_ewc 0
```
## run EWC on PMNIST
```
python run.py --experiment pmnist
              --approach ewc
              --model alexnet
              --epochs 50
              --batch 128
              --lr 0.025
              --lamb 0.0003
              --lamb_ewc 20000
```
## run Progressive Network on PMNIST
```
python run.py --experiment pmnist
              --approach progressive
              --model alexnet
              --epochs 50
              --batch 128
              --lr 0.025
              --lamb 0.0003
```
## run Learn to Grow on PMNIST
```
python run.py --experiment pmnist
              --approach ltg
              --model alexnet
              --epochs 50
              --o_epochs 50
              --batch 128
              --o_batch 128
              --lr 0.025
              --lamb 0.0003
              --o_lr 0.025
              --o_lr_a 0.0003
              --o_lamb 0.0003
              --o_lamb_a 0.001
              --o_lamb_size 0
```
## run SEU on PMNIST
```
python run.py --experiment pmnist
              --approach seu
              --model auto
              --search_layers 4
              --eval_layers 5
              --epochs 50
              --c_epochs 100
              --o_epochs 100
              --batch 128
              --c_batch 512
              --o_batch 128
              --lr 0.025
              --lamb 0.0003
              --c_lr 0.025
              --c_lr_a 0.01
              --c_lamb 0.0003
              --o_lr 0.025
              --o_lr_a 0.01
              --o_lamb 0.0003
```
