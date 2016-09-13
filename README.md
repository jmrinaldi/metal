Metal is a very simple wrapper that allows you to train neural networks in torch easily. For instance, setting up some data, as well as creating, training, and evaluating a binary classifier takes ten lines:

```
local metal = require 'metal' 

local x_tr = torch.randn(1000,10)
local x_te = torch.randn(1000,10)
local y_tr = torch.ge(torch.sum(x_tr,2),0):double()
local y_te = torch.ge(torch.sum(x_te,2),0):double()

local net  = nn.Sequential():add(nn.Linear(10,1)):add(nn.Sigmoid())
local ell  = nn.BCECriterion()

for i=1,100 do 
  metal.train(net,ell,x_tr,y_tr,parameters)
  print(i,metal.eval(net,ell,x_te,y_te,parameters))
end
```
