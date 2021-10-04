import torch
from torch.nn import PairwiseDistance, BatchNorm1d, Dropout, Sigmoid, LeakyReLU, Linear, Module, ReLU, Sequential

class MarginalEnergy(Module):
    def __init__(self, input_dim, layers):
        super(MarginalEnergy, self).__init__()
        dim = input_dim
        seq = []
        for item in list(layers):
            seq += [Linear(dim, item), LeakyReLU(0.2)]
            dim = item
        self.seq = Sequential(*seq)
        self.last = None
        self.final = Sequential(*[Linear(dim, 1), LeakyReLU(.2)])

    def forward(self, input):
        self.last = self.seq(input)
        energy = self.final(self.last)
        return energy

    def get_last(self):
        return self.last

class ConditionalEnergy(Module):
    def __init__(self, input_dim, marg_dim, layers):
        super(ConditionalEnergy, self).__init__()
        seq = [Linear(input_dim, layers[0]), LeakyReLU(0.2)]
        self.seq_cond = Sequential(*seq)
        seq = [Linear(marg_dim, layers[0]), LeakyReLU(0.2)]
        self.seq_marg = Sequential(*seq)
        dim = layers[0] + layers[0]
        seq = []
        for item in list(layers)[1:]:
            seq += [Linear(dim, item), LeakyReLU(0.2)]
            dim = item
        seq += [Linear(dim, 1), LeakyReLU(0.2)]
        self.seq = Sequential(*seq)

    def forward(self, input, marg_last):
        energy = self.seq(torch.cat([self.seq_cond(input),self.seq_marg(marg_last)],dim=-1))
        return energy

class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = LeakyReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim, batchnorm=False):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        #seq.append(LeakyReLU())
        #seq.append(Sigmoid())
        self.seq = Sequential(*seq)
        self.bn = BatchNorm1d(data_dim,affine=False)
        self.do_bn = batchnorm

    def forward(self, input):
        #data = torch.exp(-self.seq(input))*2-1
        #data = self.seq(input)*2-1
        #data = self.bn(self.seq(input))
        data = self.seq(input)
        if self.do_bn:
            data = self.bn(data)
        return data
