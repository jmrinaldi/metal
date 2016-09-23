local metal = require 'metal' 
local optim = require 'optim'

local x_tr = torch.randn(1000,10) -- training inputs
local x_te = torch.randn(1000,10) -- test inputs
local y_tr = torch.ge(torch.sum(x_tr,2),0):double() -- training labels
local y_te = torch.ge(torch.sum(x_te,2),0):double() -- test labels

local net  = nn.Sequential():add(nn.Linear(10,1)):add(nn.Sigmoid()) -- network
local ell  = nn.BCECriterion() -- loss function

-- optional parameters
local parameters = {
  gpu = false,
  verbose = false,
  batchSize = 16,
  optimizer = optim.adam,
  optimState = {
    beta1 = 0.5
  }
}

for i=1,100 do  -- For 100 epochs...
  -- train on training data
  metal.train(net,ell,x_tr,y_tr,parameters)  
  -- print loss and accuracy on test data
  print(i,metal.eval(net,ell,x_te,y_te,parameters))
end
