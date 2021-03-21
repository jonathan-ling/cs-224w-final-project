# Augmenting Node Information for Homogeneous Graphs

The code in this repo includes experiments done with augmenting node information for homogeneous graphs. The baseline model is based off of GCN code from [Kipf and Welling](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/ddi). 

The main parameter we added was embedding_or_feature_type. This can be of type 'xavier', 'he', 'uniform', 'hops', 'cycles', 'subgraphs', 'multifeature',  or 'xavier_multifeature' as mentioned in our paper. 

## He
```
python gnn.py --embedding_or_feature_type he
```

## Xavier Multifeature
```
python gnn.py --embedding_or_feature_type xavier_multifeature
```

## Reference Performance

<img src='https://user-images.githubusercontent.com/47932450/111916982-d454de00-8a3a-11eb-8019-dcd860ac33c9.png'/>
