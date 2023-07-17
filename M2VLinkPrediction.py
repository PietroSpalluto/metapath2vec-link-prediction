from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from torch_geometric.nn.models import MetaPath2Vec
from torch_geometric.transforms import RandomLinkSplit

import torch

import numpy as np

import copy

from M2VecCustom import M2VecCustom


class M2VLinkPrediction:
    def __init__(self, data, link_type, rev_link_type, metapath, embedding_dim, walk_length,
                 context_size, walks_per_node):
        self.data = data
        self.clf = None
        self.link_type = link_type
        self.rev_link_type = rev_link_type
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.metapath = metapath
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.context_size = context_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MetaPath2Vec(self.train_data.edge_index_dict,
                                 embedding_dim=self.embedding_dim,
                                 metapath=self.metapath,
                                 walk_length=self.walk_length,
                                 context_size=self.context_size,
                                 walks_per_node=self.walks_per_node,
                                 num_negative_samples=5,
                                 sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=0)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

        self.best_score = 0
        self.best_model = None
        self.best_clf = None

    def split_data(self):
        transform = RandomLinkSplit(edge_types=self.link_type,
                                    rev_edge_types=self.rev_link_type,
                                    add_negative_train_samples=True,
                                    neg_sampling_ratio=1,
                                    disjoint_train_ratio=0)
        train_data, val_data, test_data = transform(self.data)
        print('Train data:')
        print(train_data)
        print('Validation data:')
        print(val_data)
        print('Test data:')
        print(test_data)
        return train_data, val_data, test_data

    def train_embedding(self, epoch, log_steps=100):
        print('Training embedding...')
        self.model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(self.loader):
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            # preds = self.model(self.link_type[0], self.link_type[2],
            #                    pos_rw, neg_rw)
            # loss = self.model.loss(preds, self.train_data[self.link_type].edge_label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(self.loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0

    def evaluate_embedding(self):
        self.model.eval()

        # one = self.model(self.link_type[0],
        #                  self.train_data[self.link_type].edge_label_index[0])
        # two = self.model(self.link_type[2],
        #                  self.train_data[self.link_type].edge_label_index[1])
        # p = torch.sigmoid(torch.sum(one * two, dim=-1)).cpu().detach().numpy()
        # pred_label = np.zeros_like(p, dtype=np.int64)
        # pred_label[np.where(p > 0.5)[0]] = 1
        # pred_label[np.where(p <= 0.5)[0]] = 0
        # acc = np.sum(pred_label == self.train_data[self.link_type].edge_label) / len(pred_label)

        score = self.run_link_prediction_model()
        if score > self.best_score:
            self.best_model = copy.deepcopy(self.model)
            self.best_clf = copy.deepcopy(self.clf)

        return score

    def test_embedding(self):
        print('Testing classifier...')
        self.best_model.eval()

        score = self.evaluate_link_prediction_model(self.best_model, self.best_clf, self.test_data)

        return score

    def train_link_prediction_model(self, model, max_iter, binary_operator='l1'):
        print('Training link prediction model...')
        lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
        self.clf = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])
        link_features = []

        if binary_operator == 'l1':
            one = self.model(self.link_type[0],
                             self.train_data[self.link_type].edge_label_index[0])
            two = self.model(self.link_type[2],
                             self.train_data[self.link_type].edge_label_index[1])
            p = torch.sigmoid(torch.sum(one * two, dim=-1)).cpu().detach().numpy()
            pred_label = np.zeros_like(p, dtype=np.int64)
            pred_label[np.where(p > 0.5)[0]] = 1
            pred_label[np.where(p <= 0.5)[0]] = 0
            acc = np.sum(pred_label == self.train_data[self.link_type].edge_label) / len(pred_label)

            link_features = self.link_examples_to_features(model,
                                                           self.train_data[self.link_type].edge_label_index,
                                                           self.operator_l1,
                                                           self.link_type[0],
                                                           self.link_type[2])
        elif binary_operator == 'l2':
            link_features = self.link_examples_to_features(model,
                                                           self.train_data[self.link_type].edge_label_index,
                                                           self.operator_l2,
                                                           self.link_type[0],
                                                           self.link_type[2])

        self.clf.fit(link_features, self.train_data[self.link_type].edge_label)

    def evaluate_link_prediction_model(self, model, clf, data, binary_operator='l1'):
        print('Evaluating link prediction model...')
        link_features_test = []
        if binary_operator == 'l1':
            link_features_test = self.link_examples_to_features(model,
                                                                data[self.link_type].edge_label_index,
                                                                self.operator_l1,
                                                                self.link_type[0],
                                                                self.link_type[2])
        elif binary_operator == 'l2':
            link_features_test = self.link_examples_to_features(model,
                                                                data[self.link_type].edge_label_index,
                                                                self.operator_l2,
                                                                self.link_type[0],
                                                                self.link_type[2])

        score = self.evaluate_roc_auc(clf, link_features_test, data[self.link_type].edge_label)

        return score

    def run_link_prediction_model(self, max_iter=2000):
        self.train_link_prediction_model(self.model, max_iter)
        score = self.evaluate_link_prediction_model(self.model, self.clf, self.val_data)

        return score

    @staticmethod
    def link_examples_to_features(model, edge_label_index, binary_operator, head_type, tail_type):

        return [
            binary_operator(model(head_type, batch=torch.tensor(src))[0].detach().numpy(),
                            model(tail_type, batch=torch.tensor(dst))[0].detach().numpy())
            for src, dst in zip(edge_label_index.tolist()[0], edge_label_index.tolist()[1])
        ]

    @staticmethod
    def evaluate_roc_auc(clf, link_features, link_labels):
        predicted = clf.predict_proba(link_features)

        # check which class corresponds to positive links
        positive_column = list(clf.classes_).index(1)
        return roc_auc_score(link_labels, predicted[:, positive_column])

    @staticmethod
    def operator_l1(u, v):
        return np.abs(u - v)

    @staticmethod
    def operator_l2(u, v):
        return (u - v) ** 2
