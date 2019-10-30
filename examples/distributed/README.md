## Single Machine Multi GPU Training

Use PyTorch Distributed Data Parallelism with NCCL backend.

### Run

First run the graph server:

```sh
python run_graph_server.py --dataset reddit-self-loop --num-workers 3
```

Second run the client and train the model on multi GPUs:

```sh
DGLBACKEND=pytorch python gcn_ns_ddp.py --gpu 0,1,2 --dataset reddit-self-loop --num-neighbors 10 --batch-size 30000 --test-batch-size 30000
```