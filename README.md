# Dynamic Knowledge Routing Network For Target-Guided Open-Domain Conversation

This repository contains the source code for the following paper:

[Dynamic Knowledge Routing Network For Target-Guided Open-Domain Conversation](https://arxiv.org/abs/2002.01196)  
Jinghui Qin, Zheng Ye, Jianheng Tang, Xiaodan Liang; AAAI 2020

## Model Overview
![DKRN](./DKRN.jpg)

## Prerequisites
Create virtural environment (recommended):
```shell
conda create -n TGODC python=3.6
source activate TGODC
```

Install the required packages:
```shell
pip install -r requirements.txt
```

Install Texar locally:
```shell
cd texar-0.2.1
pip install .
```

Note: make sure that your environment has installed **cuda 10.0**.

## Usage

### Data Preparation
We conduct our experiments on the Target-Guided PersonaChat Dataset(TGPC) proposed by ([Tang et al. 2019](https://arxiv.org/abs/1905.11553)), and our proposed Chinese Weibo Conversation Dataset(CWC).  
Download the processed data([TGPC](https://drive.google.com/open?id=1Q4pRpFsxap2vqZ83mmMBpTNPnYpPxLHT), [CWC](https://drive.google.com/open?id=1NYBLxkLnGRNv720SLIcQyX7Um6rbxsAc)) and unzip them into the root directory of this code repository.

For the data preprocessing details, you could see the code inside the `preprocess` and `preprocess_weibo` directories.

### Turn-level Supervised Learning

In this project, we propose DKRN agent with more accurate next-topic prediction, compared with 5 different types of baseline agents (kernel/neural/matrix/retrieval/retrieval_stgy).
You can modify the configration of each agent in the `config` directory for the TGPC dataset and `config_weibo` directory for the CWC dataset.

To train our DKRN dialogue agent, you need to first train the keyword prediction module, and then train the response retrieval module:
```shell
python train.py --mode train_kw --agent neural_dkr
python train.py --mode train --agent neural_dkr
python train.py --mode test --agent neural_dkr
```

To train the baseline agents, change the `--agent` argument with a specific agent name (kernel/neural/matrix/).

Specially, for the retrieval and retrieval_stgy agents, you only need to train the response retrieval module:
```shell
python train.py --mode train --agent retrieval
python train.py --mode test --agent retrieval
```
Note: the retrieval agent and the retrieval_stgy agent share the same retrieval module. You only need to train one of them.

In addition, we provide a checkpoint of our DKRN agent trained on TGPC dataset. If you wanna use it, download the [DKRN checkpoint](https://drive.google.com/open?id=1OSkxSyWrMH_AanbWq4GodO896ll1DfDo) and unzip it into the `save/` directory(if `save` directory doesn't exist in the root directory, just create a directory named `save`).

### Target-guided Conversation

After turn-level training, you can start target-guided conversation with our DKRN agent.
```shell
python target_chat.py --agent neural_dkr --times 3
```

You can also watch the simulation of the target-guided conversation between the retrieval agent pretending the user and our DKRN agent. The success rate and average turns would be shown in the end.

```shell
python target_simulation.py --agent neural_dkr --times 500 --dataset TGPC --print_details=False --use_fixed_start_corpus=True
```

Note:
1. Baseline agents (kernel/neural/matrix/retrieval/retrieval_stgy) are also supported by changing the `--agent` argument.
2. Set `--times` argument to indicate the conversation times or the simulation times.