# Simple MANN (using LSTM)
Simple MANN is a meta-learning implementation inspired by the paper OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING. Even though the name suggests MANN(Memory Augmented Neural Networks), instead of NTM this implementation uses a single layer LSTM as memory unit. It is evaluated based on the Omniglot Dataset.

# Setup
### Requirements
Create virtualenv and install the requirements using `pip install -r requirements.txt`

### Data
In order to run the experiment you need to download the data from: https://github.com/brendenlake/omniglot/tree/master/python. See [`readme`](data/readme.md) for more details.

# Run
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
### Extra
Run the load_data.py to visualize the meta-learning problem presented to the network
```
python load_data.py
```
Data for 1-shot(1 row for query set) 5-way meta-learning problem with batch size of 1(for better visualization)
![Sample data](sample_data.png?raw=true "Data Loader sample data")
# Result
Accuracy vs Iteration graph for experimentation on 1-shot 5-way with batch size 16
![Accuracy vs Iteration](simple_mann.png?raw=true "Simple MANN")