-------------------------------------------------------------------------------
-- metal, easily train and evaluate neural networks in torch
--
-- Copyright (c) 2016 Facebook (David Lopez-Paz)
-- 
-- All rights reserved.
-- 
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
-- 
-- 1. Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
-- 
-- 2. Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
-- 
-- 3. Neither the names of NEC Laboratories American and IDIAP Research
--    Institute nor the names of its contributors may be used to endorse or
--    promote products derived from this software without specific prior
--    written permission.
-- 
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
-- AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
-- ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
-- LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
-- CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
-- SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
-- INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
-- CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
-- ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
-- POSSIBILITY OF SUCH DAMAGE.
-- 
-------------------------------------------------------------------------------

local metal = {}

local nn = require 'nn'
local xlua = require 'xlua'
local optim = require 'optim'

function metal.train(net, ell, x, y, parameters)
  local parameters = parameters or {}                 -- the parameters are:
  local gpu = parameters.gpu or false                 -- use GPU? 
  local verbose = parameters.verbose or false         -- display progress?
  local batchSize = parameters.batchSize or 64        -- batch size
  local optimizer = parameters.optimizer or optim.sgd -- optimizer 
  local optimState = parameters.optimState or {}      -- optimizer state

  -- first call... 
  if (net.p == nil) then
    if gpu then net:cuda() end
    if gpu then ell:cuda() end
    net.p, net.dp = net:getParameters()
  end
  
  -- set net in training mode
  net:training()
 
  -- nn.ClassNLLCriterion does not work with 2D targets
  if (ell.__typename == 'nn.ClassNLLCriterion') then
    y = y:view(-1)
  end

  -- these are the "global" input/target variables
  local input, target

  -- optimize function handle
  local function handle(x)
     net.dp:zero()
     local prediction = net:forward(input)
     local loss = ell:forward(prediction, target)
     local gradient = ell:backward(prediction, target)
     net:backward(input, gradient)
     return loss, net.dp 
  end

  -- random permutation 
  local p = torch.randperm(x:size(1))
  -- proceed in minibatches
  for i=1,x:size(1),batchSize do
    -- collect random minibatch
    local to = math.min(i+batchSize-1,x:size(1))
    local idx = p[{{i,to}}]:long()
    input  = x:index(1,idx)
    target = y:index(1,idx)
    if gpu then input = input:cuda() end
    if gpu then target = target:cuda() end 
    -- train
    optimizer(handle, net.p, optimState)
    -- report progress if verbose 
    if (verbose == true) then
      xlua.progress(i,x:size(1))
    end
  end
end

function metal.eval(net, ell, x, y, parameters)
  local parameters = parameters or {}
  local batchSize = parameters.batchSize or 16
  local gpu = parameters.gpu or false
  local verbose = parameters.verbose or false

  -- first call... 
  if (net.p == nil) then
    if gpu then net:cuda() end
    if gpu then ell:cuda() end
    net.p, net.dp = net:getParameters()
  end

  net:evaluate()

  -- nn.ClassNLLCriterion does not work with 2D targets
  if (ell.__typename == 'nn.ClassNLLCriterion') then
    y = y:view(-1)
  end

  local nBatches = 0
  local accuracy = 0
  local loss = 0
    
  for i=1,x:size(1),batchSize do
    local to = math.min(i+batchSize-1,x:size(1))
    local input  = x[{{i,to}}]
    local target = y[{{i,to}}]
    if gpu then input = input:cuda() end
    if gpu then target = target:cuda() end 
    local prediction = net:forward(input)
    local batchLoss = ell:forward(prediction, target)
    nBatches = nBatches + 1
    loss = loss + batchLoss

    if (ell.__typename == 'nn.BCECriterion') then
      local plabels = torch.ge(prediction,0.5):long()
      accuracy = accuracy + torch.eq(plabels,target:long()):double():mean()
    end

    if ((ell.__typename == 'nn.ClassNLLCriterion') or
        (ell.__typename == 'nn.CrossEntropyCriterion')) then
      local _, plabels = torch.max(prediction,2)
      accuracy = accuracy + torch.eq(plabels,target:long()):double():mean()
    end
    if (verbose == true) then
      xlua.progress(i,x:size(1))
    end
  end

  return loss / nBatches, accuracy / nBatches
end

function metal.save(net, fname)
  net:evaluate()
  net:clearState()
  net.p = nil
  net.dp = nil
  torch.save(fname, net)
end

function metal.load(fname)
  return torch.load(fname)
end

return metal
