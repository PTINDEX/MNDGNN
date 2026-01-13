import torch
from torch.nn import ModuleList, Linear
from torch import nn, optim
from utils.data_utils import row_l1_normalize, tau_softmax, get_norm_adj
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MNDGCNConv(torch.nn.Module):
  def __init__(self, input_dim, output_dim, alpha, mask, deg_enc=None, num_graph_types=2, net_types=6, tau=0.0):
    super(MNDGCNConv, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.alpha = alpha
    self.mask = mask
    self.num_graph_types = num_graph_types

    self.lin_src_to_dst = Linear(input_dim, output_dim)
    self.lin_dst_to_src = Linear(input_dim, output_dim)
    self.in_filter = Linear(input_dim, 1)
    self.out_filter = Linear(input_dim, 1)
    self.fc = Linear(input_dim, output_dim)
    self.in_degrees = []
    self.out_degrees = []

    global_max_in_degree = 0
    global_max_out_degree =0

    if deg_enc is not None:
      for i in range(net_types):
        in_degree = deg_enc[i][0].long()
        out_degree = deg_enc[i][1].long()
        self.in_degrees.append(in_degree)
        self.out_degrees.append(out_degree)
        global_max_in_degree = max(global_max_in_degree, int(max(in_degree)))
        global_max_out_degree = max(global_max_out_degree, int(max(out_degree)))

      self.in_deg_enc = nn.Embedding(global_max_in_degree + 1, input_dim, padding_idx=0)
      self.out_deg_enc = nn.Embedding(global_max_out_degree + 1, input_dim, padding_idx=0)

    self.adj_norms = [None] * net_types
    self.adj_t_norms = [None] * net_types
    self.I = None
    self.bn = nn.BatchNorm1d(output_dim)
    self.tau = tau

  def dirichlet_energy(self, adj, x):
    x1 = row_l1_normalize(x)
    e = (self.I + adj) @ torch.multiply(x1, x1) - 2 * (torch.multiply(((self.I + adj) @ x1), x1) - torch.multiply(x1, x1))
    return e
  
  def deg_filter(self, x, mask, i):
    e_out = self.dirichlet_energy(self.adj_norms[i].to(device), x.to(device))
    e_in = self.dirichlet_energy(self.adj_t_norms[i].to(device), x)

    encoded = self.out_deg_enc(self.out_degrees[i])
    dimention =encoded.shape[1]
    
    C_out = self.out_filter((-e_out + self.out_deg_enc(self.out_degrees[i])))
    C_in = self.in_filter((-e_in + self.in_deg_enc(self.in_degrees[i])))

    effective_tau = torch.exp(self.tau) + 0.1
    C = tau_softmax(torch.cat((C_out, C_in), 1), log_tau=effective_tau)
    C_out = C[:, 0].unsqueeze(1).float()
    C_in = C[:, 1].unsqueeze(1).float()
    C_out = torch.multiply(C_out, mask["out_deg_mask"].unsqueeze(1)) +mask["out_deg_mask_bias"].unsqueeze(1)
    C_in = torch.multiply(C_in, mask["in_deg_mask"].unsqueeze(1)) +mask["in_deg_mask_bias"].unsqueeze(1)
    return C_out, C_in, dimention
  
  def forward(self, x, edge_indices, edge_types):
    if self.I is None:
      row = torch.arange(0, x.shape[0] - 1).long().to(device)
      self.I = SparseTensor(row=row, col=row, sparse_sizes=(x.shape[0], x.shape[0]))
    out = torch.zeros(x.size(0), self.output_dim).to(x.device)
    C_ins, C_outs = [], []
    out_list = []

    for i, (edge_index, edge_type) in enumerate(zip(edge_indices,edge_types)):
      if self.adj_norms[i] is None:
        row, col = edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(x.shape[0], x.shape[0]))
        self.adj_norms[i] = get_norm_adj(adj, norm="dir")
        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(x.shape[0], x.shape[0]))
        self.adj_t_norms[i] = get_norm_adj(adj_t, norm="dir")
      out_nei = (self.adj_norms[i].cpu() @ x.cpu()).to(device)
      in_nei = (self.adj_t_norms[i].cpu() @ x.cpu()).to(device)
      C_out, C_in, dimention = self.deg_filter(x, self.mask[i], i)

      out += torch.multiply(C_out, self.lin_src_to_dst(out_nei)) + torch.multiply(C_in, self.lin_dst_to_src(in_nei))
      out_list.append(torch.multiply(C_out, self.lin_src_to_dst(out_nei)) + torch.multiply(C_in, self.lin_dst_to_src(in_nei)))
      
      C_ins.append(C_in)
      C_outs.append(C_out)
    out = out / len(edge_indices)
    C_ins = torch.mean(torch.stack(C_ins), dim=0)
    C_outs = torch.mean(torch.stack(C_outs), dim=0)

    x2 = F.pad(x, (0, dimention - x.shape[1]))
    output = out + self.alpha * self.fc(x2)
    output = self.bn(output)
    return output, [C_ins, C_outs]
  

def get_conv(conv_type, input_dim, output_dim, alpha, mask, deg_enc=None, num_graph_types=2, net_types=6, tau=0.0):
    if conv_type == "gcn":
      return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
      return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
      return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "mndgcn":
      return MNDGCNConv(input_dim, output_dim, alpha, mask, deg_enc, num_graph_types, net_types, tau)
    else:
      raise ValueError(f"Convolution type {conv_type} not supported")


class MNDGNN(torch.nn.Module):
  def __init__(
      self,
      num_features,
      num_classes,
      hidden_dim,
      num_layers=2,
      dropout=0,
      conv_type='mndgcn',
      jumping_knowledge=False,
      normalize=False,
      alpha=1/2,
      learn_alpha=False,
      mask=None,
      deg_enc=None,
      num_graph_types=2,
      net_types=6,
      num_nodes=10013):
    super(MNDGNN, self).__init__()

    self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
    self.tau = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    self.num_layers = num_layers
    self.dropout = dropout
    self.jumping_knowledge = jumping_knowledge
    self.normalize = normalize

    self.Z = nn.Parameter(torch.empty(num_nodes, hidden_dim))
    self.M = nn.Bilinear(hidden_dim, hidden_dim, 1)
    nn.init.xavier_uniform_(self.Z)
    nn.init.xavier_uniform_(self.M.weight)
    self.M.bias.data.zero_()

    output_dim = hidden_dim if jumping_knowledge else num_classes
    if num_layers == 1:
      self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha, mask, deg_enc, num_graph_types, net_types, self.tau)])
    else:
      self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha, mask, deg_enc, num_graph_types, net_types, self.tau)])
      for _ in range(num_layers - 2):
        self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha, mask, deg_enc, num_graph_types, net_types, self.tau))
      self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha, mask, deg_enc, num_graph_types, net_types, self.tau))
    
    if jumping_knowledge is not None:
      input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
      self.lin = Linear(input_dim, num_classes)
      self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

  def forward(self, x, edge_indices, edge_types):
    pos_hs, neg_hs, gs =[], [], []
    xs = []
    C_ins, C_outs = [], []

    for i, conv in enumerate(self.convs):
      if i==0:
        pos_h, [C_in, C_out] = conv(x, edge_indices, edge_types)
      else:
        pos_h, [C_in, C_out] = conv(pos_h, edge_indices, edge_types)
      if i != len(self.convs) - 1 or self.jumping_knowledge:
        pos_h = F.relu(pos_h)
        pos_h = F.dropout(pos_h, p=self.dropout, training=self.training)
        if self.normalize:
          pos_h = F.normalize(pos_h, p=2, dim=1)
      pos_hs.append(pos_h)

      if i==0:
        neg_h, _ = conv(x[torch.randperm(x.size(0))], edge_indices, edge_types)
      else:
        neg_h, _ = conv(neg_h[torch.randperm(neg_h.size(0))], edge_indices, edge_types)
      if i != len(self.convs) - 1 or self.jumping_knowledge:
        neg_h = F.relu(neg_h)
        neg_h = F.dropout(neg_h, p=self.dropout, training=self.training)
        if self.normalize:
          neg_h = F.normalize(neg_h, p=2, dim=1)
      neg_hs.append(neg_h)

      gs.append(pos_h.mean(dim=0, keepdim=True))
      xs += [pos_h]
      C_ins.append(C_in)
      C_outs.append(C_out)

    C_ins = torch.mean(torch.stack(C_ins), dim=0)
    C_outs = torch.mean(torch.stack(C_outs), dim=0)

    if self.jumping_knowledge is not None:
      x1 = self.jump(xs)
      x1 = self.lin(x1)
    return torch.nn.functional.log_softmax(x1, dim=1), C_ins, C_outs, pos_hs, neg_hs, gs
      

def get_model(args, mask, deg_enc=None, net_types=6, num_nodes=10013):
  return MNDGNN(
    num_features=args.num_features,
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    num_classes=args.num_classes,
    dropout=args.dropout,
    conv_type=args.conv_type,
    jumping_knowledge=args.jk,
    normalize=args.normalize,
    alpha=args.alpha,
    learn_alpha=args.learn_alpha,
    mask=mask,
    deg_enc=deg_enc,
    num_graph_types=2,
    net_types=net_types,
    num_nodes=num_nodes
  )


class LightingFullBatchModelWrapper(pl.LightningModule):
  def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None, beta=0.5):
    super().__init__()
    self.model = model
    self.lr = lr
    self.weight_decay = weight_decay
    self.evaluator = evaluator
    self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask
    self.beta = beta
  
  def forward(self, x, edge_index, edge_types):
    out, _, _, _, _, _= self.model(x, edge_index, edge_types)
    return out
  
  def evaluate(self, y_pred, y_true):
    y_pred = y_pred[:, 1].detach()
    y_pred_label = (y_pred > 0.5).long().squeeze()
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_pred_label_np = y_pred_label.cpu().numpy()
    accuracy = accuracy_score(y_true_np, y_pred_label_np)
    auc_score = roc_auc_score(y_true_np, y_pred_np.squeeze())
    precision, recall, _ = precision_recall_curve(y_true_np, y_pred_np.squeeze())
    aupr = auc(recall, precision)
    f1 = f1_score(y_true_np, y_pred_label_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_label_np)
    return accuracy, auc_score, aupr, f1, mcc

  def training_step(self, batch):
    with torch.no_grad():
      self.model.tau.data.clamp_(-2.3, 1.6)

    x, y = batch['x'], batch['y']
    edge_indices, edge_types = batch['edge_indices'], batch['edge_types']
    out, C_ins, C_outs, _, _, _ = self.model(x, edge_indices, edge_types)

    # current_tau = math.exp(self.model.tau.item())
    # self.log("tau_value", current_tau)

    loss_sup = F.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
    loss_reg = (torch.sum((C_ins - C_ins.mean())**2) + torch.sum((C_outs - C_outs.mean())**2))
    loss = loss_sup + self.beta * loss_reg
    self.log('train_loss', loss)
    val_loss = F.nll_loss(out[self.val_mask], y[self.val_mask].squeeze())
    self.log('val_loss', val_loss + self.beta * loss_reg) 

    y_pred = torch.exp(out)
    train_acc, train_auc, train_aupr, train_f1, train_mcc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
    self.log('train_acc', train_acc)
    self.log('train_auc', train_auc)
    self.log('train_aupr', train_aupr)
    self.log('train_f1', train_f1)
    self.log('train_mcc', train_mcc)
    val_acc, val_auc, val_aupr, val_f1, val_mcc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
    self.log('val_acc', val_acc)
    self.log('val_auc', val_auc)
    self.log('val_aupr', val_aupr)
    self.log('val_f1', val_f1)
    self.log('val_mcc', val_mcc)
    return loss
  
  def test_step(self, batch, batch_idx):
    x, y = batch['x'], batch['y']
    edge_indices, edge_types = batch['edge_indices'], batch['edge_types']
    out = self.model(x, edge_indices, edge_types)
    y_pred = torch.exp(out[0])
    test_acc, test_auc, test_aupr, test_f1, test_mcc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
    self.log('test_acc', test_acc)
    self.log('test_auc', test_auc)
    self.log('test_aupr', test_aupr)
    self.log('test_f1', test_f1)
    self.log('test_mcc', test_mcc)

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    return optimizer


def train_mngcontrast(model, optimizer, x, edge_indices, edge_types):
    model.train()
    optimizer.zero_grad()
    
    _, _, _, pos_hs, neg_hs, gs = model(x, edge_indices, edge_types)
    loss = 0
    for pos_h, neg_h, g in zip(pos_hs, neg_hs, gs):
        g = g.expand_as(pos_h)
        loss += -torch.log(model.M(pos_h, g).sigmoid() + 1e-15).mean()
        loss += -torch.log(1 - model.M(neg_h, g).sigmoid() + 1e-15).mean()
    
    pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
    neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)
    reg_loss = (model.Z - pos_mean).pow(2).sum() - (model.Z - neg_mean).pow(2).sum()
    total_loss = loss + 0.001 * reg_loss
    
    total_loss.backward()
    optimizer.step()
    return total_loss.item()