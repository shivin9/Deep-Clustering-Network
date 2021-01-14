### DCN: Deep Clustering Network

Forked from xuyxu (https://github.com/xuyxu/Deep-Clustering-Network)

### Results


| NMI | ARI | parameters |
|-----|-----|------------|
| 0.793 | 0.676 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.01 --lr 0.001 |
| 0.758 | 0.647 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.05 --lr 0.001|
| 0.748 | 0.629 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.05 |
| 0.737 | 0.618 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.05 --lr 0.0001|
| 0.737 | 0.595 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.05 --lr 0.0005 |
| 0.701 | 0.581 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 |
| 0.627 | 0.412 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.05 --lr 0.005 | 
| 0.678 | 0.526 | mnist.py --latent-dim 10 --epoch 50 --pre-epoch 50 --lamda 0.1 --lr 0.001|
|-----|-----|------------|
|0.81|0.73| *original paper claim:* pre-/eps 50, lamda 0.05, 4-layer 500-500-2000-10|


