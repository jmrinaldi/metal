package = "metal"
version = "1.0-0"

source = {
   url = "git://github.com/lopezpaz/metal.git"
}

description = {
   summary = "Train and evaluate neural networks easily in torch.",
   homepage = "https://github.com/lopezpaz/metal",
   license = "BSD"
}

dependencies = {
   "torch",
   "xlua",
   "optim",
   "nn"
}

build = {
   type = "builtin",
   modules = {
      ['metal.init'] = 'init.lua'
   }
}
