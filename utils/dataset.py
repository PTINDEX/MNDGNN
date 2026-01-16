from torch.utils.data import Dataset
import yaml

# Dataset Interface
class MultiGraphFullBatchDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return 1
  
  def __getitem__(self, index):
    return {'x':self.data['gene'].x,
            'edge_indices':self.data.edge_indices,
            'edge_types':self.data.edge_types,
            'mask':self.data.mask if hasattr(self.data, 'mask') else None,
            'y':self.data['gene'].y if 'y' in self.data['gene'] else None}
 
  
def use_best_hyperparams(args, dataset_name):
  best_params_file = "Best_hyperparams.yml"
  with open(best_params_file, "r") as file:
    hyperparams = yaml.safe_load(file)
  for name, value in hyperparams[dataset_name].items():
    if hasattr(args, name):
      setattr(args, name, value)
    else:
      raise ValueError(f"Trying to set non exsiting parameter: {name}")
  return args