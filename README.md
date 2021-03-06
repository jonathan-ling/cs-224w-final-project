# Augmenting Node Information for Homogeneous Graphs

The code in this repo includes experiments done with augmenting node information for homogeneous graphs. The baseline model is based off of GCN code from [Kipf and Welling](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/ddi). 


## Abstract 
Graph Neural Network (GNN) aggregation schemes involve recursively aggregating representations of network nodes, but popular methods such as mean-pooling in Graph Convolutional Networks (GCNs) and max-pooling in GraphSAGE are not able to adequately distinguish some basic graph structures, including symmetrical ones. In our paper, we use the link prediction task on the drug-drug interaction dataset from Open Graph Benchmark to analyze the impacts of adding different types of information. We investigate the usage of different random initializations of embeddings, which are updated across epochs, static node information that is constant across epochs, and pre-trained model embeddings. Compared to using a uniform distribution for the embeddings, we determine that a combination of binarized k-hop, cycle, and subgraph features with static Xavier initialization performs significantly better than the baseline. Further work with better initializations could significantly improve existing models. 

The main parameter we added was embedding_or_feature_type. This can be of type 'xavier', 'he', 'uniform', 'hops', 'cycles', 'subgraphs', 'multifeature',  or 'xavier_multifeature' as mentioned in our paper. 

## He
```
python gnn.py --embedding_or_feature_type he
```

## Xavier Multifeature
```
python gnn.py --embedding_or_feature_type xavier_multifeature
```

## Framework

<img src='https://user-images.githubusercontent.com/47932450/111917049-25fd6880-8a3b-11eb-9d9b-cdeaf86b74ad.png'/>
<em>Initializing embeddings and node features using various randomization methods and node characteristics respectively</em>


## Reference Performance

<img src='https://user-images.githubusercontent.com/47932450/111916982-d454de00-8a3a-11eb-8019-dcd860ac33c9.png'/>
