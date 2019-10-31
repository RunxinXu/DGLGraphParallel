import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from DGLGraphParallel.graph_parallelv2 import DGLGraphDataParallel, DGLNodeFlowLoader
import os

class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNSampling(nn.Module):
  def __init__(self,
               in_feats,
               n_hidden,
               n_classes,
               n_layers,
               activation,
               dropout):
    super(GCNSampling, self).__init__()
    self.n_layers = n_layers
    if dropout != 0:
        self.dropout = nn.Dropout(p=dropout)
    else:
        self.dropout = None
    self.layers = nn.ModuleList()
    # input layer
    self.layers.append(NodeUpdate(in_feats, n_hidden, activation))
    # hidden layers
    for i in range(1, n_layers):
        self.layers.append(NodeUpdate(n_hidden, n_hidden, activation))
    # output layer
    self.layers.append(NodeUpdate(n_hidden, n_classes))

  def forward(self, nf):
    nf.layers[0].data['activation'] = nf.layers[0].data['features']

    for i, layer in enumerate(self.layers):
      h = nf.layers[i].data.pop('activation')
      if self.dropout:
          h = self.dropout(h)
      nf.layers[i].data['h'] = h
      nf.block_compute(i,
                       fn.copy_src(src='h', out='m'),
                       lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                       layer)

    h = nf.layers[-1].data.pop('activation')
    return h


class GCNInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCNInfer, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, test=True))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, test=True))
        # output layer
        self.layers.append(NodeUpdate(n_hidden, n_classes, test=True))

    def forward(self, nf):
      nf.layers[0].data['activation'] = nf.layers[0].data['features']

      for i, layer in enumerate(self.layers):
          h = nf.layers[i].data.pop('activation')
          nf.layers[i].data['h'] = h
          nf.block_compute(i,
                           fn.copy_src(src='h', out='m'),
                           fn.sum(msg='m', out='h'),
                           layer)
      h = nf.layers[-1].data.pop('activation')
      return h


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

    features = torch.FloatTensor(data.features)
    #labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.sum().item()
    n_val_samples = val_mask.sum().item()
    n_test_samples = test_mask.sum().item()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))
    g = DGLGraph(data.graph, readonly=True)

    # prepare model
    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        F.relu,
                        args.dropout)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    infer_model = GCNInfer(in_feats,
                   args.n_hidden,
                   n_classes,
                   args.n_layers,
                   F.relu)

    # data parallel
    if args.gpu != 'cpu':
      device = torch.device("cuda:0")
      model.to(device)
      model = DGLGraphDataParallel(model)
      infer_model.to(device)
    
    train_loader = DGLNodeFlowLoader(data, 
                                     args.batch_size,
                                     args.n_layers+1,
                                     train_nid,
                                     sample_type='neighbor',
                                     num_neighbors=args.num_neighbors,
                                     num_worker=32)
    # start training
    epoch_dur = []
    for epoch in range(args.n_epochs):
      step = 0
      batch_dur = []
      time_epoch = time.time()
      
      model.train()
      for inputs, label in train_loader:
        t0 = time.time()
        # forward
        pred = model(inputs)
        loss = loss_fcn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_dur.append(time.time() - t0)
        step += 1
        if step % 1 == 0:
          print('step [{}]: batch time(s): {:.4f}'.format(step, batch_dur[-1]))
            
      epoch_dur.append(time.time() - time_epoch)
      print('Epoch [{}]: Average Batch Time(s): {:.4f}, Epoch Time: {:.4f}'.format(epoch, np.mean(batch_dur), epoch_dur[-1]))
      print('Average Epoch Time(s): {:.4f}'.format(np.mean(epoch_dur[3:])))

    # validation -- on single GPU
    for infer_param, param in zip(infer_model.parameters(), model.module.parameters()):    
      infer_param.data.copy_(param.data)
    num_acc = 0.
    for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                         g.number_of_nodes(),
                                                         neighbor_type='in',
                                                         num_workers=32,
                                                         num_hops=args.n_layers+1,
                                                         seed_nodes=test_nid):
      nf.copy_from_parent()
      infer_model.eval()
      with torch.no_grad():
        pred = infer_model(nf)
        batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
        batch_labels = labels[batch_nids]
        num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

    print("Test Accuracy {:.4f}".format(num_acc / n_test_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=str, default='cpu',
            help="gpu ids. such as 0 or 0,1,2")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)


