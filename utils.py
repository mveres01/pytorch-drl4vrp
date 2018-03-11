import os
import numpy as np
import torch


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    from graphviz import Digraph

    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def gen_dataset(task, num_train, num_val, data_dir='data'):

    # Task specific configuration - generate dataset if needed
    COP, size = task.split('_')

    size = int(size)
    data_dir_ = os.path.join(data_dir, COP)

    if COP == 'sort':

        from tasks import sorting as env

        input_dim = 1

        train_name = env.create_dataset(num_train, data_dir, size, 'train')
        train = env.SortingDataset(train_name)

        val_name = env.create_dataset(num_val, data_dir, size, 'train')
        valid = env.SortingDataset(val_name)

    elif COP == 'tsp':

        from tasks import tsp as env

        input_dim = 2

        _ = env.create_dataset(problem_size=str(size), data_dir=data_dir_)

        train = env.TSPDataset(train=True, size=size, num_samples=num_train)
        valid = env.TSPDataset(train=True, size=size, num_samples=num_val)
    else:
        raise Exception('Task %s not supported' % COP)

    return input_dim, train, valid
