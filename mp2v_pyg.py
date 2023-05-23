import numpy as np
import pandas as pd

from DatasetLoader import DatasetLoader
from KnowledgeGraph import KnowledgeGraph
from M2VLinkPrediction import M2VLinkPrediction

from sklearn.preprocessing import LabelEncoder

import torch

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

data_path = ''

dataset_loader = DatasetLoader(data_path)
edge_index = dataset_loader.load_data()

le = LabelEncoder()
le.fit(np.concatenate((edge_index['node1'], edge_index['node2'])))

edge_index['node1id'] = le.transform(edge_index['node1'])
edge_index['node2id'] = le.transform(edge_index['node2'])
len(le.classes_)
edge_attr_dict = {'gene-drug': 0, 'gene-gene': 1, 'bait-gene': 2, 'gene-phenotype': 3, 'drug-phenotype': 4}
edge_index['typeid'] = edge_index['type'].apply(lambda x: edge_attr_dict[x])
edge = torch.tensor(edge_index[['node1id', 'node2id']].values, dtype=torch.long)
edge_attr = torch.tensor(edge_index['typeid'].values, dtype=torch.long)

kg = KnowledgeGraph(edge_index, edge_attr_dict)
kg.triples()

data = kg.edge_index
gene = data[data['type'] == 'gene-phenotype']['node1id'].drop_duplicates()
drug = data[data['type'] == 'drug-phenotype']['node1id'].drop_duplicates()
phenotype = pd.concat([data[data['type'] == 'gene-phenotype']['node2id'], data[data['type'] == 'drug-phenotype']['node2id']], ignore_index=True).drop_duplicates().dropna()
gene_phenotype = torch.Tensor([data[data['type'] == 'gene-phenotype']['node1id'],
                               data[data['type'] == 'gene-phenotype']['node2id']]).to(torch.long)
phenotype_gene = torch.Tensor([data[data['type'] == 'gene-phenotype']['node2id'],
                               data[data['type'] == 'gene-phenotype']['node1id']]).to(torch.long)
drug_phenotype = torch.Tensor([data[data['type'] == 'drug-phenotype']['node1id'],
                               data[data['type'] == 'drug-phenotype']['node2id']]).to(torch.long)
phenotype_drug = torch.Tensor([data[data['type'] == 'drug-phenotype']['node2id'],
                               data[data['type'] == 'drug-phenotype']['node1id']]).to(torch.long)

heterodata = HeteroData()
heterodata['gene'].x = torch.Tensor(gene).to(torch.long)
heterodata['drug'].x = torch.Tensor(drug).to(torch.long)
heterodata['phenotype'].x = torch.Tensor(phenotype).to(torch.long)
heterodata['gene', 'interacts', 'phenotype'].edge_index = gene_phenotype
heterodata['drug', 'interacts', 'phenotype'].edge_index = drug_phenotype
heterodata['phenotype', 'interacts2', 'gene'].edge_index = phenotype_gene
heterodata['phenotype', 'interacts2', 'drug'].edge_index = phenotype_drug

print(heterodata)

transform = RandomLinkSplit(edge_types=('gene', 'interacts', 'phenotype'),
                            rev_edge_types=('phenotype', 'interacts2', 'gene'),
                            add_negative_train_samples=True,
                            neg_sampling_ratio=1,
                            disjoint_train_ratio=0)
train_data, val_data, test_data = transform(heterodata)

print('Train data:')
print(train_data)
print('Validation data:')
print(val_data)
print('Test data:')
print(test_data)

metapath = [('gene', 'interacts', 'phenotype'),
            ('phenotype', 'interacts2', 'drug'),
            ('drug', 'interacts', 'phenotype'),
            ('phenotype', 'interacts2', 'gene')]
link_type = ('gene', 'interacts', 'phenotype')
rev_link_type = ('phenotype', 'interacts2', 'gene')

link_pred_model = M2VLinkPrediction(heterodata, link_type, rev_link_type, metapath, embedding_dim=128,
                                    walk_length=50, context_size=4, walks_per_node=5)

for epoch in range(1, 5):
    link_pred_model.train_embedding(epoch)
    acc = link_pred_model.test_embedding()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
