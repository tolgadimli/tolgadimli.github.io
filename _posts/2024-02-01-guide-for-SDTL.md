## Guide for Using Synchronous Distributed Training Library (SDTL)

### What does the SDTL library include?

The SDTL stands for Synchronous Distributed Training Library for Data Parallel training of Deep Neural models. It includes Parameter-Sharing algorithms LSGD and EASGD as well as Pytorch’s vanilla Data Parallel training DistributedDataParallel(DDP) which are all synchronous. DDP works by exchanging and averaging gradient vectors among the workers in the distributed training environment. In Parameter-Sharing however, the workers in the distributed training environment transfer their local information by exchanging their model parameters and forming a consensus model after aggregation. This is quite different than how vanilla DDP and it brings some advantages. Further explanations and details about how these two types of Data Parallel training work and their comparison can be found here. If you also would like to learn more details specifically about the Parameter-Sharing based EASGD and LSGD algorithms see this awesome blogpost written by Yunfei Teng.

### How to use it?
The use of our published library is fairly straightforward. Typically, one uses the following code snippet to set up PyTorch’s Distributed Data Parallel training environment.
```python
model     = torch.nn.parallel.DistributedDataParallel(model.to(device))
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.mom, weight_decay=args.wd, nesterov=True)
```

Our code requires only an additional line to initialize CommOptimizer object that works as a wrapper around the (local) optimizer.

```python
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.mom, weight_decay=args.wd, nesterov=True)
optimizer = CommOptimizer(optimizer, dist_optim_name=args.dist_optim_name, world_size = world_size, comm_period=args.comm_period, dist_pulling_strength=args.c, local_pulling_strength=args.p)
```
As you can see, it is that simple to initialize a Parameter-Sharing based Data Parallel training using our library!

### More Details About CommOptimizer and DistOptimizer Classes
We first have the class `CommOptimizer` which combines both the local optimizer and the distributed optimization steps. For initialization, it expects a torch.optim.Optimizer object as the local optimizer, name of the distributed optimizer (EASGD | LSGD) and a bunch of other arguments that are required for initialization of the distributed optimizer (which are explained in the list below). It has `step()` and `zero_grad()` functions similar to torch.optim.Optimizer class. Because it wraps the local optimizer, the `zero_grad()` function simply zeroes out the calculated gradient values of the local optimizer as before. The step function takes the local optimization step and it carries out the distributed optimization step if the distributed optimizer is specified.

Then, we have the `DistOptimizer` class which is a base class for the EASGD and LSGD classes. It takes the following arguments:

- `local_optimizer (torch.optim.Optimizer)`: the to fetch the parameter groups and make use of the gradient information if needed
- `world_size (int)`: number of participating workers in the distributed training environment (i.e. total number of GPUs)
- `local_pulling_strength (float, optional)`: strength of the pulling force taken toward the most recent center model
- `dist_pulling_strength: (float)`: strength of the pulling force taken toward the center model during the distributed optimization phase 
- `comm_period (int)`: the period that determines after how many local iterations should the workers initiate communication and enter the distributed optimization phase

The `DistOptimizer` class has several functions. 

The `local_pulling_step` pulls the worker, with strength `local_pulling_strength`, toward the center model that is constructed with the most recent distributed optimization phase. Note that this should not be confused with the local optimizer (torch.optim.Optimizer) step as it does not involve any gradient calculation. It benefits the training by allowing for more exploration of the loss landscape with applying a force toward an informative, consensus direction.

`dist_train_step` is an auxiliary function that can be utilized to collect some statistics of the training, the gradient information from the local optimizer etc. It can also be seen as optional if no such step is required.

`comm_train_step`: This function initiates the communication between all the workers in the distributed training environment. In accordance with the specified distributed optimizer name, it aggregates the collected model parameters and forms a consensus variable/center model. After forming the center model, this step also pulls all the workers toward it. The strength of the pulling force is specified with the dist_pulling_strength argument.

`step`: This function combines `local_pulling_step`, `dist_train_step` and `comm_train_step` functions as part of the distributed optimization step. Notice that, among these three, the `comm_train_step` function is the only function that requires inter-worker communication for parameter exchange in the distributed training environment for building the consensus model. Also, both the `local_pulling_step` and `comm_train_step` functions update to the worker’s model parameters whereas `dist_train_step` does not. The `step` function also has a counter that keeps track of the local number of iterations. `comm_train_step` function is executed only when the counter reaches the comm_period and the counter is reset after the execution.



