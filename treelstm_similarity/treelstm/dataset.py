import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree


# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))

        self.ltrees = self.read_trees(os.path.join(path, 'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path, 'b.parents'))

        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (ltree, lsent, rtree, rsent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')
        return labels

class QuoraDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(QuoraDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = 2

        self.lsentences = self.read_sentences(os.path.join(path,'1','sents_proc.toks'))
        self.rsentences = self.read_sentences(os.path.join(path,'2','sents_proc.toks'))

        self.ltrees = self.read_trees(os.path.join(path,'1','dparents.txt'))
        self.rtrees = self.read_trees(os.path.join(path,'2','dparents.txt'))

        self.labels = self.read_labels(os.path.join(path,'labels.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        if (len(lsent)<40 and len(rsent)<40):
            return (ltree, lsent, rtree, rsent, label)
        else:
            return self.__getitem__((index+1)%self.size)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        with open(filename,'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees
    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root
    # def read_tree(self, line):
    #     parents = map(int,line.split())
    #     trees = dict()
    #     root = None
    #     for i in range(1, len(parents)+1):
    #         #if not trees[i-1] and parents[i-1]!=-1:
    #         if i-1 not in trees.keys() and parents[i-1]!=-1:
    #             idx = i
    #             prev = None
    #             while True:
    #                 parent = parents[idx-1]
    #                 if parent == -1:
    #                     break
    #                 tree = Tree()
    #                 if prev is not None:
    #                     tree.add_child(prev)
    #                 trees[idx-1] = tree
    #                 tree.idx = idx-1
    #                 #if trees[parent-1] is not None:
    #                 if parent-1 in trees.keys():
    #                     trees[parent-1].add_child(tree)
    #                     break
    #                 elif parent==0:
    #                     root = tree
    #                     break
    #                 else:
    #                     prev = tree
    #                     idx = parent
    #     return root

    # def read_labels(self, filename):
    #     with open(filename,'r') as f:
    #         labels = map(lambda x: float(x), f.readlines())
    #         labels = torch.Tensor(labels)
    #     return labels

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')
        return labels
