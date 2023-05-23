# LARA

## Abstract 
As Graph Neural Networks (GNNs) have been widely used in real-world applications, model explanations are required not only by users but also by legal regulations. However, simultaneously achieving high fidelity and low computational costs in generating explanations has been a challenge for current methods. In this work, we propose a framework of GNN explanation named LeArn Removal-based Attribution (LARA) to address this problem. Specifically, we introduce removal-based attribution and demonstrate its substantiated link to interpretability fidelity theoretically and experimentally. The explainer in LARA learns to generate removal-based attribution which enables providing explanations with high fidelity. A strategy of subgraph sampling is designed in LARA to improve the scalability of the training process. In the deployment, LARA can efficiently generate the explanation through a feed-forward pass. We benchmark our approach with other state-of-the-art GNN explanation methods on six datasets. Results highlight the effectiveness of our framework regarding both efficiency and fidelity. In particular, LARA is 3.5 times faster and achieves higher fidelity than the state-of-the-art method on the large dataset ogbn-arxiv (more than 160K nodes and 1M edges), showing its great potential in real-world applications.

## Run this repo
### Prepare the environment
Use the command to install the environment for running this repo
```
conda env create -f environment.yml
```

### Train LARA
Train LARA on ogbn-arxiv by running the command:
```
python train_lara.py --dataset='ogbn-arxiv' --type_model=GraphSAINT --num_splits=10 --random_seed=21  --batch_size=16
```

A trained target model can be found in the folder ``saved_models``. Us the following model to train a target model, e.g., 
```
python train_target_model.py --cuda_num=0  --type_model=GraphSAINT --dataset=ogbn-arxiv

```

### Test LARA
Test LARA on ogbn-arxiv by running:
```
python test_lara.py --dataset='ogbn-arxiv' --type_model=GraphSAINT --num_splits=10 --random_seed=21  --batch_size=1024
```

