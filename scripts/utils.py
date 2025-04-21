# -*- coding: utf-8 -*-
# File              : utils.py
# Author            : Joy
# Date              : 2024/06/09
# Last Modified Date: 2024/08/09
# Last Modified By  : Joy
# Reference         :
# Description       : tools for data processing, evaluations, etc
# ******************************************************

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler

def compute_binary_result(prediction_probs, y):
    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score  # precision_recall_curve
    result = {}
    prediction_y = (prediction_probs > 0.5).astype(int)
    result['acc'] = accuracy_score(y, prediction_y)
    result['prec'] = precision_score(y, prediction_y)
    result['auc'] = roc_auc_score(y, prediction_probs)
    result['recall'] = recall_score(y, prediction_y)
    result['f1_scores'] = 2*result['recall']*result['prec']/(result['recall']+result['prec'])
    return result

def plot_box(df):
        ax = sns.boxplot(data = df, edgecolor='black')
        ax = sns.swarmplot(data = df)
        ax.set_ylabel('Test Accuracy')
        plt.show()

def extract_embeddings(dataloader, model):
    import torch
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size