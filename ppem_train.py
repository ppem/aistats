import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from models import ConditionalEnergy, MarginalEnergy, Generator
from kernel import LinearRBF
from gof import LinearKSD

def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def calc_gradient_penalty_margcond(E_c, E_m, real_data, fake_data_c, fake_data_m, device):
    alpha = torch.rand(real_data.shape[0], 1).to(device)
    alpha_m = alpha.expand(real_data[:,idx_m].size())
    alpha_c = alpha.expand(real_data[:,idx_c].size())

    interpolates_m = alpha_m * real_data[:,idx_m] + ((1 - alpha_m) * real_data[:,idx_m])
    interpolates_m = interpolates_m.to(device)
    interpolates_m = autograd.Variable(interpolates_m, requires_grad=True)

    interpolates_c = alpha_c * real_data[:,idx_c] + ((1 - alpha_c) * fake_data_c)
    interpolates_c = interpolates_c.to(device)
    interpolates_c = autograd.Variable(interpolates_c, requires_grad=True)

    disc_interpolates_m = E_m(interpolates_m)
    disc_interpolates = E_c(interpolates_c,interpolates_m)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates_c,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# train the marginal generator and energy model adversarialy
def marg_train():
    for epoch in range(epochs):
        for batch_num, real in enumerate(dataloader):
            real_m=real[:,idx_m]
            curr_batch_size = real.shape[0]

            # clear gradients
            G_m.zero_grad()
            E_m.zero_grad()

            # sample batch from latent space
            latent = torch.randn([curr_batch_size,latent_dim_m]).to(device)

            # generate fake data
            fake = G_m(latent)

            # update the generator
            loss_G = E_m(fake).mean() 
            loss_G.backward()
            opt_G_m.step()

            # clear energy gradient
            E_m.zero_grad()

            # energy model update iterations
            for i in range(e_iter):
                latent = torch.randn([curr_batch_size,latent_dim_m]).to(device)
                fake = G_m(latent)
                loss_E = relu(margin - E_m(fake)).mean() + E_m(real_m).mean() + grad_lambda*calc_gradient_penalty(E_m,real_m,fake,device)                
                ksd_stat = ksd.test_stat(autograd.Variable(real_m,requires_grad=True))*curr_batch_size
                loss_E_pen = loss_E + ksd_mult*relu(test_thresh - ksd_stat)
                loss_E_pen.backward()
                if sigma:
                    torch.nn.utils.clip_grad_norm_(E_m.parameters(), clip)
                    for name, param in E_m.named_parameters():
                        param.grad = param.grad + torch.FloatTensor(param.grad.size()).normal_(0, sigma * clip).to(device)
                opt_E_m.step()
        print('Epoch {} of {}'.format(epoch,epochs))


# train the conditional generator and energy model adversarialy
def cond_train():
    for epoch in range(epochs):
        for batch_num, real in enumerate(dataloader):
            curr_batch_size = real.shape[0]

            # clear gradients
            G_c.zero_grad()
            E_c.zero_grad()

            # sample batch from latent space
            latent_c = torch.randn([curr_batch_size,latent_dim_c]).to(device)

            # generate fake data
            fake_c = G_c(torch.cat([latent_c,real[:,idx_m]],dim=-1))

            # update the generator
            loss_G = E_c(fake_c,real[:,idx_m]).mean()
            loss_G.backward()
            opt_G_c.step()

            # clear energy gradient
            E_c.zero_grad()

            # energy model update iterations
            for i in range(e_iter):
                latent_c = torch.randn([curr_batch_size,latent_dim_c]).to(device)
                fake_c = G_c(torch.cat([latent_c,real[:,idx_m]],dim=-1))
                loss_E = relu(margin-E_c(fake_c,real[:,idx_m])).mean() + E_c(real[:,idx_c],real[:,idx_m]).mean() + grad_lambda*calc_gradient_penalty_margcond(E_c,E_m,real,fake_c,fake_m,device)
                loss_E.backward()
                if sigma:
                    torch.nn.utils.clip_grad_norm_(E_c.parameters(), clip)
                    for name, param in E_c.named_parameters():
                        param.grad = param.grad + torch.FloatTensor(param.grad.size()).normal_(0, sigma * clip).to(device)
                opt_E_c.step()
        print('Epoch {} of {}'.format(epoch,epochs))


# path to datafile
filename = 'file.npy'

# indices of private variables 
idx_m = torch.Tensor([0,1]).long()

# indices of nonprivate variables 
idx_c = torch.Tensor([2,3]).long()
        
# hyperparameters
margin = 10
batch_size = 128
epochs = 100
latent_dim_m = 128
latent_dim_c = 128
lr_g = 5e-4
lr_e = 1e-3
grad_lambda=10
e_iter=5
ksd_mult=1
test_thresh=3.841
clip = 1
sigma = 0
relu=nn.ReLU()


# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the dataloader
dataset = torch.from_numpy(np.load(filename)).float().to(device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

# samples, features
n = dataset.shape[0]
m_c = idx_c.shape[0]
m_m = idx_m.shape[0]

# Generator
G_m = Generator(latent_dim_m,[256,256],m_m).to(device)
G_c = Generator(latent_dim_c+m_m,[256,256],m_c).to(device)

# Energy
E_m = MarginalEnergy(m_m,[256,256]).to(device)
E_c = ConditionalEnergy(m_c,m_m,[256,256]).to(device)

# KSD
ksd = LinearKSD(E_m,LinearRBF())

# Optimizers for G and D
opt_E_m = torch.optim.Adam(E_m.parameters(), lr=lr_e)
opt_E_c = torch.optim.Adam(E_c.parameters(), lr=lr_e)
opt_G_m = torch.optim.Adam(G_m.parameters(), lr=lr_g)
opt_G_c = torch.optim.Adam(G_c.parameters(), lr=lr_g)

# train marginal model
marg_train()

# train conditional model
cond_train()

# save trained models
torch.save({'energy_m':E_m.state_dict(),'energy_c':E_c.state_dict(),'generator_m':G_m.state_dict(),'generator_c':G_c.state_dict()},'models/{}.pt'.format(filename))
