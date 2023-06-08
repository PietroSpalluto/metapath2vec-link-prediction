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
        super().__init__()
        self.lp_mlp = torch.nn.Linear(self.embedding_dim*2, 2)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.reset_parameters()

    def loss(self, pos_rw: Tensor, neg_rw: Tensor, y: Tensor) -> Tensor:
        orig_loss = self.loss(pos_rw, neg_rw)
        node_one = self(pos_rw)
        preds = self.lp_mlp(node_one)
        p = torch.nn.Sigmoid(preds)
        loss = self.loss_fn(p, y)
        return orig_loss + loss
