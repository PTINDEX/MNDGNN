import torch
import numpy as np
import random
import argparse
import time
from utils.data_loading import DataIO
from utils.dataset import MultiGraphFullBatchDataset, use_best_hyperparams
from torch.utils.data import DataLoader
from utils.data_utils import degree_encoding, get_available_accelerator, get_mask
from model import get_model, LightingFullBatchModelWrapper, train_mngcontrast
from utils.data_augmentation import data_aug
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
import os
from utils.predictions_utils import save_predictions, top_predictions
from datetime import datetime
import math
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# dataset params
parser.add_argument('--dataset', type=str, default='gene', help='Name of dataset')
parser.add_argument('--network_paths',
                    default=[
                            'PPI=./datasets/PPI_PyG.txt',
                            'Complexes=./datasets/Complexes_PyG.txt',
                            'Pathway=./datasets/KEGG_PyG.txt',
                            'Regulatory=./datasets/RegNetwork_PyG.txt',
                            'DawnNet=./datasets/DawnNet_PyG.txt',
                            'Kinase=./datasets/Kinase_Substrate_PyG.txt'
                            ],
                    nargs='+',
                    metavar='KEY=VALUE',
                    help='Directory of network paths')
parser.add_argument('--feature_path',
                    default='./datasets/Feature_for_PyG.csv',
                    help='Directory of feature path')
parser.add_argument('--true_id_path',
                    default='./datasets/true_ids.txt',
                    help='Directory of true_ids path')
parser.add_argument('--mapping_dict',
                    default='./datasets/mapping_dict_EID_Index.pickle',
                    help='Directory of mapping dict path')
parser.add_argument('--kcdg',
                    default='./datasets/kcdg_intersec_EID.csv',
                    help='Know cancer driver genes')
parser.add_argument('--maybe_cdg',
                    default='./datasets/maybe_cdg.txt',
                    help='Maybe cancer diver genes')
parser.add_argument('--directed_flags', type=dict,
                    default={'PPI':False, 'Complexes':False, 'Pathway':True, 'Regulatory':True , 'DawnNet':True, 'Kinase':True},
                    help='Dictionary indicating which networks are directed')
parser.add_argument('--checkpoint_directory', type=str, default='checkpoint', help='Directory to save checkpoints')
# model params
parser.add_argument('--model', type=str, default='mndgnn', help='Model type')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of model')
parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.0, help='Feature dropout')
parser.add_argument('--alpha', type=float, default=0.5, help='Direction convex combination params')
parser.add_argument('--learn_alpha', action='store_true')
parser.add_argument('--conv_type', type=str, default='mndgcn', help='Convolution type')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--jk', type=str, choices=['max', 'cat', None], default='max', help='Jumping Konwledge')
parser.add_argument('--beta', type=float, default=0.5, help='Direction convex combination params')
# training params
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--num_epochs', type=int, default=100, help='Max number of epochs')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--num_runs', type=int, default=1, help='Max number of runs')
# system params
parser.add_argument('--gpu_idx', type=int, default=0, help='Indexes of gpu to run program on')
parser.add_argument('--profiler', action='store_true')
parser.add_argument('--num_workers', type=int, default=0, help='Num of workers for the dataloader')

now = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
print('------' + now + '------')
print('------ Loading Data ------')
args = parser.parse_args()

def run(args):
  seed = 2025
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True

  network_paths = {}
  if args.network_paths:
    for item in args.network_paths:
      key, value = item.split('=')
      network_paths[key] = value

  dataIO = DataIO()
  data, net_types = dataIO.load_network(args.feature_path, need_features=True, directed_flag=args.directed_flags, **network_paths)
  print('data')
  print(data)
  print('----------')
  EID_Index = dataIO.load_mapping_dict(args.mapping_dict)
  kcdg = dataIO.load_konw_cdg(args.kcdg)
  maybe_cdg = dataIO.load_maybe_cdg(args.maybe_cdg)
  Index_EID = {v:k for k, v in EID_Index.items()}

  data_loader = DataLoader(MultiGraphFullBatchDataset(data), batch_size=1, collate_fn=lambda batch:batch[0])
  batch = next(iter(data_loader))
  print(batch.keys())

  deg_enc = []
  val_accs, val_aucs, val_auprs, val_f1s, val_mccs = [], [], [], [], []
  test_accs, test_aucs, test_auprs, test_f1s, test_mccs = [], [], [], [], []  
  for num_run in range(args.num_runs):
    args.num_features = data['gene'].x.shape[1]
    args.num_classes = 2
    num_nodes = data['gene'].x.shape[0]
    for net_name in net_types:
      deg_enc.append(degree_encoding(data['gene', net_name, 'gene'].edge_index, num_nodes=num_nodes))

    print("------ Getting Model ------")
    model = get_model(args, mask=data.mask, deg_enc=deg_enc, net_types=len(net_types), num_nodes=num_nodes)
    print(model)

    # Multiplex graphs contrastive pretraining
    optimizer = torch.optim.AdamW(
    [{'params': [model.Z]},
     {'params': model.M.parameters()}],
    lr=0.01)
    for epoch in range(20):
      loss = train_mngcontrast(model.to(device), optimizer, data['gene'].x.to(device), data.edge_indices, data.edge_types)
      print(f"Epoch {epoch}: Loss={loss:.4f}")
    embeddings = model.Z.detach().cpu()

    # Data augmentation
    data = data_aug(data, kcdg, embeddings, EID_Index, maybe_cdg, args.true_id_path)

    # Five-fold cross-validation
    # test: 10%
    rnd_state = np.random.RandomState(seed)
    num_nodes = data['gene'].y.shape[0]
    labeled_nodes = torch.where(data['gene'].y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    num_test = math.floor(num_labeled_nodes * 0.1)
    idxs = list(range(num_labeled_nodes))
    rnd_state.shuffle(idxs)
    test_idx = idxs[:num_test]
    train_val_idx = np.array(idxs[num_test:])
    test_idx = labeled_nodes[test_idx]
    test_mask = get_mask(test_idx, num_nodes)

    # Performing five-fold cross-validation on the remaining 90% of the data
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx)):
      print(f'\n=== Fold {fold+1}/5 ===')
      train_idx = labeled_nodes[train_idx.tolist()]
      val_idx = labeled_nodes[val_idx.tolist()]
      train_mask = get_mask(train_idx, num_nodes)
      val_mask = get_mask(val_idx, num_nodes)
      model = get_model(args, mask=data.mask, deg_enc=deg_enc, net_types=len(net_types), num_nodes=num_nodes)

      lit_model = LightingFullBatchModelWrapper(
        model=model, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        train_mask=train_mask,
        val_mask=val_mask, 
        test_mask=test_mask, 
        evaluator=None, 
        beta=args.beta)
      
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
      early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=args.patience)
      model_summary_callback = ModelSummary(max_depth=-1)
      if not os.path.exists(f"{args.checkpoint_directory}/"):
        os.mkdir(f"{args.checkpoint_directory}/")
      model_checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        mode='max',
        dirpath=f"{args.checkpoint_directory}/{str(timestamp)}"
      )

      trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=args.num_epochs,
        callbacks=[
          early_stopping_callback,
          model_summary_callback,
          model_checkpoint_callback
        ],
        profiler='simple' if args.profiler else None,
        accelerator=get_available_accelerator(),
        devices=[args.gpu_idx]
      )

      # Model training and evaluation
      print("------ Training Model ------")
      trainer.fit(model=lit_model, train_dataloaders=data_loader)

      val_acc = trainer.callback_metrics['val_acc'].item()
      val_auc = trainer.callback_metrics['val_auc'].item()
      val_aupr = trainer.callback_metrics['val_aupr'].item()
      val_f1 = trainer.callback_metrics['val_f1'].item()
      val_mcc = trainer.callback_metrics['val_mcc'].item()

      val_accs.append(val_acc)
      val_aucs.append(val_auc)
      val_auprs.append(val_aupr)
      val_f1s.append(val_f1)
      val_mccs.append(val_mcc)

      test_metrics = trainer.test(ckpt_path='best', dataloaders=data_loader)[0]
      test_accs.append(test_metrics['test_acc'])
      test_aucs.append(test_metrics['test_auc'])
      test_auprs.append(test_metrics['test_aupr'])
      test_f1s.append(test_metrics['test_f1'])
      test_mccs.append(test_metrics['test_mcc'])

    # Print results
    print("\n=== Validation Metrics ===")
    print(f"Val Acc:   {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Val AUC:   {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"Val AUPR:  {np.mean(val_auprs):.4f} ± {np.std(val_auprs):.4f}")
    print(f"Val F1:    {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Val MCC:   {np.mean(val_mccs):.4f} ± {np.std(val_mccs):.4f}")

    print("\n=== Test Metrics ===")
    print(f"Test Acc:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"Test AUC:  {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Test AUPR: {np.mean(test_auprs):.4f} ± {np.std(test_auprs):.4f}")
    print(f"Test F1:   {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test MCC:  {np.mean(test_mccs):.4f} ± {np.std(test_mccs):.4f}")

  # Loading the best model
  best_model = LightingFullBatchModelWrapper.load_from_checkpoint(
    checkpoint_path=model_checkpoint_callback.best_model_path,
    model=model,
    lr=args.lr,
    weight_decay=args.weight_decay,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    evaluator=None,
    beta=args.beta
  )

  # Prediction
  predict_mask = (data['gene'].y == -1)
  best_model.eval()
  with torch.no_grad():
    logits = best_model(data['gene'].x.to(device), data.edge_indices, data.edge_types)
    preds = logits.argmax(dim=1)
    predictions = preds[predict_mask]
  print(f"Predicted {predict_mask.sum().item()} samples") 
  print("Prediction:", predictions)

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  save_dir = f"./outputs/{timestamp}"
  os.makedirs(save_dir, exist_ok=True)
  # Saving prediction results
  save_predictions(logits, predict_mask, Index_EID, predictions, save_dir)
  # Selecting top n cancer genes from the predictions
  top_predictions(50, logits, predict_mask, Index_EID, save_dir)


if __name__ == '__main__':
  args = use_best_hyperparams(args, args.dataset)
  run(args)