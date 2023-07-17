from typing import Dict, List, Optional, Tuple

import torch.nn
from torch_geometric.nn.models import MetaPath2Vec
from torch_geometric.typing import EdgeType, NodeType, OptTensor, SparseTensor
from torch import Tensor


class M2VecCustom(MetaPath2Vec):
    def __init__(self,
                 edge_index_dict: Dict[EdgeType, Tensor],
                 embedding_dim: int,
                 metapath: List[EdgeType],
                 walk_length: int,
                 context_size: int,
                 walks_per_node: int = 1,
                 num_negative_samples: int = 1,
                 num_nodes_dict: Optional[Dict[NodeType, int]] = None,
                 sparse: bool = False, ):
        super().__init__(edge_index_dict, embedding_dim, metapath, walk_length, context_size)
        self.lp_mlp = torch.nn.Linear(self.embedding_dim*2, 2)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.reset_parameters()

    def _pos_sample(self, batch: Tensor) -> Tensor:
        print('asd')
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            adj = self.adj_dict[keys]
            batch = sample(adj, batch, num_neighbors=1,
                           dummy_idx=self.dummy_idx).view(-1)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        # rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    # def loss(self, pos_rw: Tensor, neg_rw: Tensor, y: Tensor) -> Tensor:
    #     orig_loss = self.loss(pos_rw, neg_rw)
    #     node_one = self(pos_rw)
    #     preds = self.lp_mlp(node_one)
    #     p = torch.nn.Sigmoid(preds)
    #     loss = self.loss_fn(p, y)
    #     return orig_loss + loss


def sample(src: SparseTensor, subset: Tensor, num_neighbors: int,
           dummy_idx: int) -> Tensor:

    mask = subset < dummy_idx
    rowcount = torch.zeros_like(subset)
    rowcount[mask] = src.storage.rowcount()[subset[mask]]
    mask = mask & (rowcount > 0)
    offset = torch.zeros_like(subset)
    offset[mask] = src.storage.rowptr()[subset[mask]]

    rand = torch.rand((rowcount.size(0), num_neighbors), device=subset.device)
    rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(offset.view(-1, 1))

    col = src.storage.col()[rand]
    col[~mask] = dummy_idx
    return col
