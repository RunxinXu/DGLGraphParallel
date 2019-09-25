import torch
from itertools import chain
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parallel.data_parallel import _check_balance
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index

from dgl import NodeFlow
from dgl.contrib.sampling import NeighborSampler

class DGLNodeFlowLoader():
  def __init__(self, graph, labels, batch_size,
               num_hops, seed_nodes, sample_type='neighbor',
               num_neighbors=8, num_worker=32):
    self.graph = graph
    self.labels = labels
    self.batch_size = batch_size
    self.type = sample_type
    self.num_hops = num_hops
    self.seed_nodes = seed_nodes
    self.num_neighbors = num_neighbors
    self.num_worker = num_worker

    self.device_num = torch.cuda.device_count()
    if self.device_num == 0:
      self.device_num = 1 # cpu
    per_worker_batch = int(self.batch_size / self.device_num)
    if self.type == "neighbor":
      self.sampler = NeighborSampler(self.graph,
                                     per_worker_batch,
                                     self.num_neighbors,
                                     neighbor_type='in',
                                     shuffle=True,
                                     num_workers=self.num_worker,
                                     num_hops=self.num_hops,
                                     seed_nodes=self.seed_nodes,
                                     prefetch=True)
    else:
      self.sampler = None
      raise RuntimeError("Currently only support Neighbor Sampling")
    
    self.sampler_iter = None
  
  def __iter__(self):
    self.sampler_iter = iter(self.sampler)
    return self

  def __next__(self):
    nf_list = []
    label_list = []
    for i in range(self.device_num):
      try:
        nf = next(self.sampler_iter)
        batch_nids = nf.layer_parent_nid(-1).to(device=self.labels.device, dtype=torch.long)
        nf_list.append(nf)
        label_list.append(self.labels[batch_nids])
      except StopIteration:
        if len(nf_list) == 0:
          raise StopIteration
        else: # the last batch
          break
    labels = torch.cat(label_list)
    return nf_list, labels


class DGLGraphDataParallel(torch.nn.Module):

  def __init__(self, module, device_ids=None, output_device=None, dim=0):
    super(DGLGraphDataParallel, self).__init__()

    if not torch.cuda.is_available():
      self.module = module
      self.device_ids = []
      return
    
    if device_ids is None:
      device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
      output_device = device_ids[0]
    
    self.dim = dim
    self.module = module
    self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    self.output_device = _get_device_index(output_device, True)
    self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
      self.module.cuda(device_ids[0])
  

  def forward(self, inputs, **kwargs):
    """
    inputs should be a list when multi-gpus is enabled.
    The list length of inputs should be equal to device num.
    Each element in inputs list should be a instance of nodeflow
    """
    if not self.device_ids:
      return self.module(*inputs, **kwargs)
    
    for t in chain(self.module.parameters(), self.module.buffers()):
      if t.device != self.src_device_obj:
        raise RuntimeError("module must have its parameters and buffers "
                           "on device {} (device_ids[0]) but found one of "
                           "them on device: {}".format(self.src_device_obj, t.device))

    if not isinstance(inputs, list):
      inputs = [inputs]
    if len(self.device_ids) < len(inputs):
      raise RuntimeError("device num [{}] is not equal to inputs length [{}]"
                         .format(len(self.device_ids), len(inputs)))
    # replicate kwargs
    kwargs = scatter(kwargs, self.device_ids[:len(inputs)], 0)
    if len(self.device_ids) == 1:
      return self.module(inputs[0])
    elif isinstance(inputs[0], NodeFlow):
      # copy inputs from its parent graph (should reside in cuda:0)
      # better way for small graphs to do this is to replica parent features 
      # to all gpus and load from its own gpu
      for device_id in range(len(inputs)):
        dgl_ctx = DataCopyContext(device_type="cuda", index=self.device_ids[device_id])
        inputs[device_id].copy_from_parent(ctx=dgl_ctx)
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    return self.gather(outputs, self.output_device)

  def replicate(self, module, device_ids):
    return replicate(module, device_ids, not torch.is_grad_enabled())

  def parallel_apply(self, replicas, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
  
  def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)


class DataCopyContext:
  def __init__(self, device_type="cpu", index=None):
    self.type= device_type
    self.index = index