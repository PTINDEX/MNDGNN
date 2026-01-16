import torch
from sklearn.model_selection import train_test_split
from deepod.models.tabular import ICL
from utils.data_utils import load_ids
from utils.augmentation_util import GraphDataAugmenter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_aug(data, kcdg, embeddings, EID_Index, maybe_cdg, true_id_path):
  true_ids = load_ids(true_id_path)
  true_len = len(true_ids)
  y = torch.full((data['gene'].x.shape[0],), -1)
  y[true_ids] = 1
  data['gene'].y = y

  # positive-sample augmentation
  augmenter = GraphDataAugmenter(num_classes=2, hidden_dim=64)
  aug_mask = (data['gene'].y != -1)
  aug_data, positive_ind = augmenter.augment_data(data, aug_mask)
  
  positive_indices = (aug_data['gene'].y == 1).nonzero().squeeze()
  cancer_genes = positive_indices.cpu().tolist()
  true_len = len(cancer_genes)

  cancer_genes = kcdg[0].tolist()
  train_genes, _ = train_test_split(cancer_genes, test_size=0.9, random_state=42)
  train_indices = [EID_Index[gene] for gene in train_genes]
  embeddings = embeddings.cpu()
  x_kcdg = torch.Tensor(embeddings[train_indices, :].numpy())
  all_genes = list(EID_Index.keys())
  unlabeled_genes = list(set(all_genes) - set(cancer_genes))
  unwanted_indices = [EID_Index[gene] for gene in unlabeled_genes]
  x_unlabeled = torch.Tensor(embeddings[unwanted_indices, :].numpy())

  # negative-sample inference
  clf = ICL(lr=1e-5, verbose=3, epochs=10)
  clf.fit(x_kcdg.numpy())
  scores = clf.decision_function(x_unlabeled.numpy())
  gene_scores = dict(zip(unlabeled_genes, scores))
  filtered_scores = [(g, s) for g, s in gene_scores.items() if g not in maybe_cdg]
  filtered_scores_sorted = sorted(filtered_scores, key=lambda x: x[1], reverse=True)
  selected_negs = [g for g, _ in filtered_scores_sorted[:true_len]]
  negs_indices = [EID_Index[gene] for gene in selected_negs]
  false_ids = torch.tensor(negs_indices, dtype=torch.long)
  combined_indices = torch.cat([true_ids.to(device), positive_ind])
  y[combined_indices] = 1
  y[false_ids] = 0
  data['gene'].y = y
  return data