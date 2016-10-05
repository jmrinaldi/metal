local metal   = require 'metal'
local gnuplot = require 'gnuplot'
local optim   = require 'optim' 

local bs = 16 -- batch size
local h  = 50 -- hidden layer representation
local Dz = 5  -- latent dimensionality
local Dx = 1  -- conditioning dimensionality
local Dy = 1  -- conditioned dimensionality

torch.manualSeed(0)

-- generator(x,z)
local g = nn.Sequential():add(nn.Linear(Dx+Dz,h)):add(nn.ReLU()):add(nn.Linear(h,Dy))
g.p, g.dp = g:getParameters()

-- discriminator(x,y)
local f = nn.Sequential():add(nn.Linear(Dx+Dy,h)):add(nn.ReLU()):add(nn.Linear(h,1)):add(nn.Sigmoid())
f.p, f.dp = f:getParameters()

-- discriminator(generator(x,z)), parameters only from generator(x,z)
local id_x = nn.Sequential():add(nn.Select(2,1)):add(nn.View(-1,1))
local fg_concat = nn.ConcatTable():add(id_x):add(g)
local fg = nn.Sequential():add(fg_concat):add(nn.JoinTable(2)):add(f)
fg.p, fg.dp = g.p, g.dp

local y_real = torch.ones(bs,1)  -- labels for real data
local y_fake = torch.zeros(bs,1) -- labels for fake data
local ell    = nn.BCECriterion() -- discriminator loss is binary cross entropy

local pf = {
  optimizer = optim.adam, optimState = { learningRate = 0.001, beta1 = 0.5 }
}

local pg = {
  optimizer = optim.adam, optimState = { learningRate = 0.001, beta1 = 0.5 }
}

local function foo(x)
  return torch.cos(x):add(torch.randn(x:size()):cmul(x*0.1))
end
    
for i=1,10000 do
  local b_z  = torch.randn(bs,Dz)
  local b_x  = torch.randn(bs,Dx)
  local b_xz = torch.cat(b_x, b_z, 2)
  local b_fake = torch.cat(b_x, metal.predict(g,b_xz), 2)
  local b_real = torch.cat(b_x, foo(b_x), 2)
  -- train discriminator
  metal.train(f,ell,torch.cat(b_fake,b_real,1),torch.cat(y_fake,y_real,1),pf)
  -- train generator
  metal.train(fg,ell,b_xz,y_real,pg)
end

local plot_x = torch.randn(10000, Dx)
local plot_z = torch.randn(10000, Dz)
local plot_p = metal.predict(g,torch.cat(plot_x,plot_z,2))
gnuplot.plot({'real', plot_x:view(-1), foo(plot_x):view(-1), '+'},
             {'fake', plot_x:view(-1), plot_p:view(-1), '+'})
io.read()
