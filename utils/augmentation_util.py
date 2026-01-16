import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, k):
    super().__init__()
    self.conv1 = ChebConv(in_channels, hidden_channels, K=k, normalization='sym')
    self.conv2 = GCNConv(hidden_channels, out_channels)
    self.norm = torch.nn.BatchNorm1d(hidden_channels)

  def forward(self, x, edge_indices):
    x = self.conv1(x, edge_indices)
    x = self.norm(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_indices)
    return F.log_softmax(x, dim=1)


class GraphDataAugmenter:
  def __init__(self, num_classes, hidden_dim=48, k_pseudo=668):
    self.num_classes = num_classes
    self.k_pseudo = k_pseudo
    self.hidden_dim = hidden_dim

  def support_augmentation(self, data, train_mask):
    with torch.no_grad():
      gcn = GCN(in_channels=data['gene'].x.shape[1], hidden_channels=self.hidden_dim, out_channels=self.num_classes, k=2).to(device)
      
      edge_index = torch.cat([e.to(device) for e in data.edge_indices], dim=1).long()
      logists = gcn(data['gene'].x.to(device), edge_index)
      probs = torch.exp(logists)
      entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

      unlabeled = ~train_mask
      pseudo_indices = []
      labeled_features = data['gene'].x[train_mask].to(device)
      pos_features_np = labeled_features.cpu().numpy()
      pos_features_np = StandardScaler().fit_transform(pos_features_np)
      if len(pos_features_np) >= 2:
        spec = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42)
        labels = spec.fit_predict(pos_features_np)

        pos_centers = np.array([pos_features_np[labels == i].mean(axis=0) for i in range(2)])
        pred_pos = (torch.argmax(probs, dim=1) == 1) & unlabeled.to(device)
        candidate_indices = torch.where(pred_pos)[0]
        candidate_features = data['gene'].x.to(device)[candidate_indices].cpu().numpy()
        candidate_entropy = entropy[pred_pos]

        _, topk_indices = torch.topk(candidate_entropy, k=min(2*self.k_pseudo, len(candidate_entropy)), largest=False)
        candidate_indices = candidate_indices[topk_indices]
        candidate_features = candidate_features[topk_indices.cpu().numpy()]

        valid_indices = []
        for i, feat in enumerate(candidate_features):
          min_dist_pos = min([np.linalg.norm(feat - center) for center in pos_centers])
          pos_dists = [np.linalg.norm(f - center) for f in pos_features_np for center in pos_centers]
          mean_dist = np.mean(pos_dists)
          std_dist = np.std(pos_dists)

          if(probs[candidate_indices[i], 1] > 0.8) and (min_dist_pos < mean_dist + std_dist):
            valid_indices.append(i)
        selected = valid_indices[:min(50, len(valid_indices))]
        pseudo_indices = candidate_indices[selected].cpu().tolist()
      else:
        pseudo_indices = []
    return torch.tensor(pseudo_indices, dtype=torch.long), probs
  
  def augment_data(self, data, train_mask):
    aug_data = data.clone()
    y = data['gene'].y
    new_y = y.clone().to(device)

    pseudo_indices, pseudo_probs = self.support_augmentation(data, train_mask)
    if len(pseudo_indices) > 0:
      pseudo_labels = torch.argmax(pseudo_probs[pseudo_indices], dim=1)
      new_y[pseudo_indices] = pseudo_labels
      aug_data['gene'].y = new_y
    positive_indices = torch.where(pseudo_labels == 1)[0]
    return aug_data, positive_indices