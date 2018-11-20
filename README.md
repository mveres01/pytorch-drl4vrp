# pytorch-drl4vrp

Implementation of: Nazari, Mohammadreza, et al. "Deep Reinforcement Learning for Solving the Vehicle Routing Problem." arXiv preprint arXiv:1802.04240 (2018).

Currently, Traveling Salesman Problems and Vehicle Routing Problems are supported. See the _tasks/_ folder for details.

## Requirements:

* pytorch=0.4.1
* matplotlib

## To Run:

Run by callinig ```python trainer.py```

Tasks and complexity can be changed through the "task" and "nodes" flag:

```python trainer.py --task=vrp --nodes=10```

To Restore a checkpoint, you must specify the path to a folder that has "actor.pt" and "critic.pt" checkpoints. Sample weights can be found [here](https://drive.google.com/open?id=1wxccGStVglspW-qIpUeMPXAGHh2HsFpF)

```python trainer.py --task=vrp --nodes=10 --checkpoint=vrp20```

## Differences from paper:

* Uses a GRU instead of LSTM for the decoder network
* Critic takes the raw static and dynamic input states and predicts a reward

## TSP Sample Tours:

__Left__: TSP with 20 cities
__Right__: TSP with 50 cities

<p align="center">
  <img src="./docs/tsp20.png" width="300"/>
  <img src="./docs/tsp50.png" width="300"/>
</p>

## VRP Sample Tours:

__Left__: VRP with 10 cities + load 20
__Right__: VRP with 20 cities + load 30

<p align="center">
  <img src="./docs/vrp10.png" width="300"/>
  <img src="./docs/vrp20.png" width="300"/>
</p>

# TSP

The following masking scheme is used for the TSP:
1. If a salesman has visited a city, it is not allowed to re-visit it. 

# VRP

The VRP deals with dynamic elements (load 'L', demand 'D') that change everytime the vehicle / salesman visits a city. Each city is randomly generated with random demand in the range [1, 9]. The salesman has an initial capacity that changes with the complexity of the problem (e.g. number of nodes)

The following __masking scheme__ is used for the VRP:
1. If there is no demand remaining at any city, end the tour. Note this means that the vehicle must return to the depot to complete
2. The vehicle can visit any city, as long as it is able to fully satisfy demand (easy to modify for partial trips if needed)
3. The vehicle may not visit the depot more then once in a row (to speed up training)
4. A vehicle may only visit the depot twice or more in a row if it has completed its route and waiting for other vehicles to finish (e.g. training in a minibatch setting) 

In this project the following dynamic updates are used:
1. If a vehicle visits a city, its load changes according to: Load = Load - Demand_i, and the demand at the city changes according to: Demand_i = (Demand_i - load)+
2. Returning to the vehicle refills the vehicles load. The depot is given a "negative" demand that increases proportional to the amount of load missing from the vehicle

# Results:

This repo only implements the "Greedy" approach during test time, which selects the city with the highest probability. Tour length comparing this project to the corresponding paper is reported below. Differences in tour length may likely be optimized further through hyperparameter search, which has not been conducted here. 

|               | Paper ("Greedy") | This  |
|---------------|------------------|-------|
| TSP20         | 3.97             | 4.032 |
| TSP50         | 6.08             | 6.226 |
| TSP100        | 8.44             |       |
| VRP10 Cap 20  | 4.84             | 5.082 |
| VRP20 Cap 30  | 6.59             | 6.904 |
| VRP50 Cap 40  | 11.39            |       |
| VRP100 Cap 50 | 17.23            |       |

# Acknowledgements:

Thanks to https://github.com/pemami4911/neural-combinatorial-rl-pytorch for insight on bug with random number generator on GPU
