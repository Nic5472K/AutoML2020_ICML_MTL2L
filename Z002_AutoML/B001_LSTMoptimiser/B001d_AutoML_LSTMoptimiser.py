###===###
import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      Fnc
import  torch.optim         as      optim
from    torch.autograd      import  Variable
import  numpy               as      np
import  math
#---
from    functools           import reduce
from    operator            import mul
#---
from scipy.stats import ortho_group

###===###
# the preprocessing regime for the gradient of the loss
# as according to the original paper of
# learning to learn by gradient descent by gradient descent
def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(1)

    return torch.cat((x1, x2), 1)

###===### 
# the main thing
class RNNOptimiser(nn.Module):

    def __init__(self, model, ID, HD):
        super(RNNOptimiser, self).__init__()
        # stores a reference model therein the network
        # for the ease to transfer and update weights
        self.RefM = model

        # storing copies of the hyperparameters
        # ID for input dimension
        # HD for hidden dimension
        self.ID = ID
        self.HD = HD

        # definig the LSTM weights
        # this is a 2-layer LSTM
        #---
        # Layer 1:
        # the forward synapse has dimension 2
        # this is becasue each gradient is preprocessed as 2 values
        # the HD is 4 * HD because there are 4 gates
        self.ih_L1 = nn.Linear(2,  4 * HD)
        self.hh_L1 = nn.Linear(HD, 4 * HD)

        # introducing some biases to allow the smooth flowing
        # of information
        self.F1_bias = nn.Parameter(torch.ones(self.HD))
        self.I1_bias = nn.Parameter(-torch.ones(self.HD))

        #---
        # Layer 2:
        # the second LSTM layer has an output dimension of 4
        # 1 dimension for each gated unit
        # the dimensionality is strictly 1 dimension per gate
        # before each output of the LSTM
        # is the update towards a parameter of the base learner
        self.ih_L2 = nn.Linear(HD, 4) 
        self.hh_L2 = nn.Linear(1, 4) 

        #---
        # network initialisation
        stdv1 = 1.0 / math.sqrt(4 * HD)
        nn.init.uniform_(self.ih_L1.weight, -stdv1, stdv1)
        nn.init.uniform_(self.ih_L1.bias,   -stdv1, stdv1)
        nn.init.uniform_(self.hh_L1.weight, -stdv1, stdv1)
        nn.init.uniform_(self.hh_L1.bias,   -stdv1, stdv1)
        stdv2 = 1.0 / math.sqrt(4) 
        nn.init.uniform_(self.ih_L2.weight, -stdv2, stdv2) 
        nn.init.uniform_(self.ih_L2.bias,   -stdv2, stdv2) 
        nn.init.uniform_(self.hh_L2.weight, -stdv2, stdv2) 
        nn.init.uniform_(self.hh_L2.bias,   -stdv2, stdv2)

    #---
    # this is where we reset the RNN optimiser hidden states
    # h for hidden state
    # c for cell state
    # by resetting, we mean by removing the information off
    # the computational graph, hence we use .data
    def ROH(self, keep_states=False, model=None):

        num_layers = 2
        #---
        self.RefM.reset()
        self.RefM.copy_params_from(model)

        if keep_states:
            for i in range(num_layers):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            self.hx.append(Variable(torch.zeros(1, self.HD)))
            self.cx.append(Variable(torch.zeros(1, self.HD)))
            self.hx.append(Variable(torch.zeros(1, 1)))
            self.cx.append(Variable(torch.zeros(1, 1)))            
            for i in range(num_layers):
                self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    #---
    # here is where the heavy lifting goes
    def forward(self, x):
          
        #---
        self.hx[0] = self.hx[0].expand(x.size(0), self.hx[0].size(1))
        self.cx[0] = self.cx[0].expand(x.size(0), self.cx[0].size(1))
        Q_1     = self.hx[0]
        S_1     = self.cx[0]

        preact  = self.ih_L1(x) + self.hh_L1(Q_1)
        
        F, I, A, O = preact.chunk(4, dim = 1)
        #---    
        F_1 = torch.sigmoid(F + self.F1_bias)
        I_1 = torch.sigmoid(I + self.I1_bias)
        A_1 = torch.tanh(   A)
        O_1 = torch.sigmoid(O)

        S_1 = F_1 * S_1 +I_1 * A_1
        Q_1 = O_1 * torch.tanh(S_1)

        self.hx[0] = Q_1
        self.cx[0] = S_1
        
        #---
        x = self.hx[0]
        self.hx[1] = self.hx[1].expand(x.size(0), self.hx[1].size(1))
        self.cx[1] = self.cx[1].expand(x.size(0), self.cx[1].size(1))
        Q_2     = self.hx[1]
        S_2     = self.cx[1]

        preact  = self.ih_L2(x) + self.hh_L2(Q_2)

        F, I, A, O = preact.chunk(4, dim = 1)
        F_2 = torch.sigmoid(F)
        I_2 = torch.sigmoid(I)
        A_2 = torch.tanh(   A)
        O_2 = torch.sigmoid(O)

        S_2 = F_2 * S_2 +I_2 * A_2
        Q_2 = O_2 * torch.tanh(S_2)

        self.hx[1] = Q_2
        self.cx[1] = S_2        
        
        #---
        Q_2 = Q_2.squeeze(1)

        #---
        # rescaling the final output by 0.1
        # as according to the original paper of
        return Q_2 * 0.1

    #---
    # this is where the base learner gets updated
    def UpdateTransfer(self, CurOptimisee, imgs):

        grads = []
        
        for module in CurOptimisee.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.RefM.get_flat_params()
        flat_params = flat_params.unsqueeze(1)
        flat_grads = preprocess_gradients(torch.cat(grads))

        flat_params = flat_params.squeeze(1)

        #---
        Update = self(flat_grads)
        flat_params = flat_params + Update

        self.RefM.set_flat_params(flat_params)

        self.RefM.copy_params_to(CurOptimisee)
        return self.RefM.model      

###===###
# the reference model
class RefMode:

    def __init__(self, model):
        self.model = model
        
    def reset(self):
        
        for module in self.model.children():
            module._parameters['weight'] = Variable(
                module._parameters['weight'].data)
            module._parameters['bias'] = Variable(
                module._parameters['bias'].data)

    def get_flat_params(self):
        params = []

        for module in self.model.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):

        offset = 0

        for i, module in enumerate(self.model.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'] = flat_params[
                offset:offset + weight_flat_size].view(*weight_shape)
            module._parameters['bias'] = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)

            offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)            
