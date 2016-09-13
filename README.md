# metal for torch

**metal** is a very simple wrapper to train and evaluate neural networks in
torch easily. For instance, setting up some data, as well as creating,
training, and evaluating a neural network binary classifier takes ten lines:

```
local metal = require 'metal' 

local x_tr = torch.randn(1000,10)
local x_te = torch.randn(1000,10)
local y_tr = torch.ge(torch.sum(x_tr,2),0):double()
local y_te = torch.ge(torch.sum(x_te,2),0):double()

local net  = nn.Sequential():add(nn.Linear(10,1)):add(nn.Sigmoid())
local ell  = nn.BCECriterion()

for i=1,100 do 
  metal.train(net,ell,x_tr,y_tr)
  print(i,metal.eval(net,ell,x_te,y_te))
end
```

The functions `metal.train` and `metal.eval` accept an optional table of
advanced `parameters` to further customize training. These include:

```
local parameters = {
  gpu = false,            -- use GPU?
  verbose = false,        -- display progress bar?
  batchSize = 16,         -- minibatch size
  optimizer = optim.adam, -- optimizer 
  optimState = {          -- optimizer table of parameters
    beta1 = 0.5
  }
}

metal.train(net,ell,x_tr,y_tr,parameters)
```

Finally, you can use `metal.save(net, 'fileName.t7')` to save your model, and
`net = metal.load('fileName.t7')` to load stored models. 
