import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import time
import pickle
from utils.data_utils import mask_zero_in_out

class DataIO():
  def __init__(self):
    pass

  def load_network(self, feature_path, need_features=True, directed_flag=None, **kwargs):
    data = HeteroData()
    features = pd.read_csv(feature_path).sort_values(by='Index')
    x = np.array(features.iloc[:, 1:])
    x = torch.tensor(x, dtype=torch.float)
    if need_features == False:
      x.fill_(1)
    data['gene'].x = x

    net_types = []
    for net_name, net_path in kwargs.items():
      edge_indic = []
      is_directed = directed_flag.get(net_name, True)
      net_adj = pd.read_csv(net_path, header=None, sep='\t')
      head = torch.tensor(net_adj[0].tolist(), dtype=torch.long)
      tail = torch.tensor(net_adj[1].tolist(), dtype=torch.long)
      if is_directed:
        edge_indic.append(torch.stack([head, tail]))
      else:
        edge_indic.append(torch.stack([head, tail]))
        edge_indic.append(torch.stack([tail, head]))
      data['gene', net_name, 'gene'].edge_index = torch.cat(edge_indic, dim=1)
      net_types.append(net_name)
    edge_indices = [data['gene', net_name, 'gene'].edge_index for net_name in net_types]
    edge_types = [0 if directed_flag.get(net_name, True) else 1 for net_name in net_types]
    data.edge_indices = edge_indices
    data.edge_types = edge_types

    data.mask = [mask_zero_in_out(data['gene'].x, data['gene', net_name, 'gene'].edge_index) for net_name in net_types]
    return data, net_types
  
  def load_mapping_dict(self, mapping_dict_path):
    file = open(mapping_dict_path, "rb")
    EID_Index_dict = pickle.load(file)
    return EID_Index_dict
  
  def load_konw_cdg(self, path):
    kcdg = pd.read_csv(path, header=None)
    return kcdg
  
  def load_maybe_cdg(self, path):
    maybe_cdg = pd.read_csv(path, header=None)
    maybe_cdg_list = maybe_cdg[0].to_list()
    return maybe_cdg_list

  @torch.no_grad()
  def save_embedding(self, embedding, path="./saved_embedding"):
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    tensor_data = embedding.data
    torch.save(tensor_data, f"{path}/embedding_{now}.pt")
    return f"{path}/embedding_{now}.pt"