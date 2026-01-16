import torch
import pandas as pd
import os
import mygene

def EID_to_name(eid_list):
  mg = mygene.MyGeneInfo()
  gene_info = mg.querymany(eid_list, scopes='entrezgene', fields='symbol', species='human')
  eid_to_name = {}
  for item in gene_info:
    if 'symbol' in item:
      eid_to_name[str(item['query'])] = item['symbol']
    else:
      eid_to_name[str(item['query'])] = 'Unknown'
  return [eid_to_name[str(eid)] for eid in eid_list]


def save_predictions(logits, predict_mask, Index_EID, predictions, save_dir):
  probs = torch.exp(logits[:, 1])
  node_indices = predict_mask.nonzero().flatten().numpy()
  eids = [Index_EID[i] for i in node_indices]
  results = pd.DataFrame({
    'Name': EID_to_name(eids),
    'EID': eids,
    'Predicted_label': predictions.cpu().numpy(),
    'Probability': probs[predict_mask].cpu().numpy()
  })
  save_path = os.path.join(save_dir, 'predictions.csv')
  results.to_csv(save_path, index=False)


def top_predictions(top_num, logits, predict_mask, Index_EID, save_dir):
  probs = torch.exp(logits[:, 1])
  valid_mask = (predict_mask == 1) & (probs.cpu() > 0.5)
  valid_indices = valid_mask.nonzero().flatten()
  valid_probs = probs[valid_indices]
  top_probs, top_indices = torch.topk(valid_probs, k=min(top_num, len(valid_probs)))
  top_global_indices = valid_indices[top_indices.cpu()].cpu().numpy()
  top_eids = [Index_EID[i] for i in top_global_indices]
  results = pd.DataFrame({
    'Name': EID_to_name(top_eids),
    'EID': top_eids,
    'Probability': top_probs.cpu().numpy(),
    'Predicted_label': 1
  })
  save_path = os.path.join(save_dir, 'top_predictions.csv')
  results.to_csv(save_path, index=False)
  print('Already done!')