import torch
import numpy as np
import os
from copy import copy
from torch.utils.data import Dataset
from utils import namenum, summary, mcmc_treeprob
from ete3 import TreeNode
from itertools import combinations

def reroot(tree: TreeNode):
    for (node1, node2) in combinations([0,1,2],r=2):
        common_ancestor = tree.get_common_ancestor(tree.search_nodes(name=node1)[0], tree.search_nodes(name=node2)[0])
        if not common_ancestor.is_root():
            tree.set_outgroup(common_ancestor)
            c1 = common_ancestor.children[0]
            c2 = common_ancestor.children[1]
            tree.remove_child(common_ancestor)
            tree.add_child(c1)
            tree.add_child(c2)
    return tree

def init_tree():
    name_dict = {}
    tree = TreeNode(name=3)
    name_dict[3] = tree
    for i in [0,1,2]:
        node = TreeNode(name=i)
        tree.add_child(node)
        name_dict[i] = node
    return tree, name_dict

def is_unrooted_bifurcating(tree : TreeNode):
    for node in tree.traverse('preorder'):
        if node.is_root():
            if len(node.children) !=3:
                return False
        elif node.is_leaf():
            pass
        else:
            if len(node.children) != 2:
                return False
    return True
            

def tree_to_vector(tree: TreeNode):
    node_decisions = {}
    node_pendant = {}
    ntaxa = len(tree.get_leaves())
    for taxon in range(ntaxa-1, 2, -1):
        leaf = tree.get_leaves_by_name(name=taxon)[0]
        sister = leaf.get_sisters()[0]
        parent = leaf.up
        grandparent = parent.up
        parent.remove_child(sister)
        grandparent.remove_child(parent)
        grandparent.add_child(sister)

        node_decisions[taxon] = sister
        node_pendant[taxon] = parent
    
    assert len(tree.get_leaves()) == 3
    tree.name = 3
    
    name_dict = {3: tree}
    decisions = []
    for taxon in range(3, ntaxa):
        sister = node_decisions[taxon]
        grandparent = sister.up
        parent = node_pendant[taxon]
        grandparent.remove_child(sister)
        grandparent.add_child(parent)
        parent.add_child(sister)

        decisions.append(sister.name)
        
        current_taxon_node = name_dict[taxon]
        parent.name, current_taxon_node.name, tree.name = 2*taxon-2, tree.name, 2*taxon-1
        name_dict[parent.name] = parent
        name_dict[current_taxon_node.name] = current_taxon_node
        name_dict[tree.name] = tree

    return decisions


def vector_to_tree(vector):
    tree, name_dict = init_tree()
    for i in range(len(vector)):
        position = vector[i]
        taxon = i + 3
        new_leaf_node = TreeNode()
        new_pendant_node = TreeNode()
        anchor_node = name_dict[position]
        parent_node = anchor_node.up

        parent_node.remove_child(anchor_node)
        new_pendant_node.add_child(anchor_node)
        new_pendant_node.add_child(new_leaf_node)
        parent_node.add_child(new_pendant_node)
        
        current_taxon_node = name_dict[taxon]
        root_node = tree
        new_leaf_node.name, new_pendant_node.name, current_taxon_node.name, root_node.name = taxon, 2*taxon-2, root_node.name, 2*taxon-1
        name_dict[new_leaf_node.name] = new_leaf_node
        name_dict[new_pendant_node.name] = new_pendant_node
        name_dict[current_taxon_node.name] = current_taxon_node
        name_dict[root_node.name] = root_node

    return tree
    
def process_data(dataset, repo='emp'):
    if repo == 'emp':
        ground_truth_path, samp_size = 'data/raw_data_DS1-8/', 750001
        tree_dict_total, tree_names_total, tree_wts_total = summary(dataset, ground_truth_path, samp_size=samp_size)
        emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
        path = os.path.join('decisions', dataset,'emp_tree_freq')
    else:
        emp_tree_freq = mcmc_treeprob('data/short_run_data_DS1-8/' + dataset + '/rep_{}/'.format(repo) + dataset + '.trprobs', 'nexus')
        path = os.path.join('decisions', dataset, f'repo{repo}')

    
    wts = list(emp_tree_freq.values())
    taxa = sorted(list(emp_tree_freq.keys())[0].get_leaf_names())
    os.makedirs(path, exist_ok=True)
    decisions_tensor = []
    for tree in emp_tree_freq.keys():
        namenum(tree, taxa)
        tree_cp = tree.copy()
        tree_cp = reroot(tree_cp)
        decisions = tree_to_vector(tree_cp)
        decisions_tensor.append(decisions)
    decisions_tensor = np.array(decisions_tensor, dtype=np.int64)
    np.save(f'{path}/decisions.npy', decisions_tensor)
    np.save(f'{path}/wts.npy', wts)
    np.save(f'{path}/taxa.npy', taxa)


class DecisionData(Dataset):
    def __init__(self, dataset, repo):
        super().__init__()
        if repo == 'emp':
            self.path = os.path.join('..', 'decisions',dataset,'emp_tree_freq')
        else:
            self.path = os.path.join('..', 'decisions',dataset,f'repo{repo}')
        self.data = torch.from_numpy(np.load(os.path.join(self.path, 'decisions.npy')))
        self.wts = np.load(os.path.join(self.path, 'wts.npy'))
        self.length = len(self.wts)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.length