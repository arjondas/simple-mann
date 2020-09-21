# Simple MANN (using LSTM)
---
Simple MANN is a meta-learning implementation inspired by the paper OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING. Even though the name suggests MANN(Memory Augmented Neural Networks), instead of NTM this implementation uses a single layer LSTM as memory unit. It is evaluated based on the Omniglot Dataset.

# Setup
---
### Requirements
Create virtualenv and install the requirements using `pip install -r requirements.txt`

### Data
In order to run the experiment you need to download the data from: https://github.com/brendenlake/omniglot/tree/master/python. See [`readme`](data/README.md) for more details.

# Run
---
Default **1-shot 5-way learning with batch size 16**
```
python model.py
```
Flags:
- k-shot `--num_samples=<K>`
- n-way `--num_classes=<N>`
- batch_size `--meta_batch_size=<BATCH_SIZE>`

For example: 2-shot 7-way with batch size 10
```
python model.py --num_samples=7 --num_classes=2 --meta_batch_size=10
```

# Result
---
Accuracy vs Iteration graph for experimentation on 1-shot 5-way with batch size 16:
"Coming Soon"