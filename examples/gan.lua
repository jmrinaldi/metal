local metal   = require 'metal'
local gnuplot = require 'gnuplot'
local optim   = require 'optim'

local bs = 16   -- batch size
local h  = 50   -- hidden layer representation
local d  = 5    -- latent dimensionality
local D  = 1    -- observed dimensionality

torch.manualSeed(0)

-- generator(z)
local g = nn.Sequential():add(nn.Linear(d,h)):add(nn.ReLU()):add(nn.Linear(h,D))
g.p, g.dp = g:getParameters()

-- discriminator(x)
local f = nn.Sequential():add(nn.Linear(D,h)):add(nn.ReLU()):add(nn.Linear(h,1)):add(nn.Sigmoid())
f.p, f.dp = f:getParameters()

-- discriminator(generator(z))
local fg = nn.Sequential():add(g):add(f)
-- train only parameters from generator(z)
fg.p, fg.dp = g.p, g.dp

-- labels for real and fake data, loss for discriminator
local y_real = torch.ones(bs,1)
local y_fake = torch.zeros(bs,1)
local ell    = nn.BCECriterion()

local pf = {
  optimizer = optim.adam, optimState = { learningRate = 0.001, beta1 = 0.5 }
}

local pfg = {
  optimizer = optim.adam, optimState = { learningRate = 0.001, beta1 = 0.5 }
}
    
for i=1,5000 do
  -- generate batch
  local b_noise = torch.randn(bs,d)
  local b_fake  = metal.predict(g,b_noise)
  local b_real  = torch.cat(torch.randn(bs/2,D)-2,torch.randn(bs/2,D)+2,1):div(2)
  -- train discriminator
  metal.train(f,ell,torch.cat(b_fake,b_real,1),torch.cat(y_fake,y_real,1),pf)
  -- train generator
  metal.train(fg,ell,b_noise,y_real,pfg)
end

gnuplot.hist(metal.predict(g,torch.randn(10000,d)))
io.read()
