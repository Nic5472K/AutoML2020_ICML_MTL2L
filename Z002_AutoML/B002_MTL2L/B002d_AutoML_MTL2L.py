###===###
# This is the other 1/2 of the main script of our paper

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

###===###
# The line below is not included in its B001d counterpart
# we will be calling ortho_group for SVDising the LSTM synapses
# Our paper can be found here
# https://arxiv.org/pdf/2007.09343.pdf
# and ortho_group is related to 
# See section 3 of page 3
#   introducing SVD to the LSTM 
# See Table 1 of page 4
#   for the new neural optimiser formulation
# And see Section B.5.1 in the Appendix of page 13
#   for some extra comments
from scipy.stats import ortho_group

###===###
# The following MLP module is also a new introduction
# to our MTL2L neural optimiser
# Its referred usage can be found in
# Equations (15) and (16) on page 4
#   and the purpose is to extract features
#   to model singular values of the SVD
# See Section B.5.2 and B.5.3 in the Appendix on page 13
#   for further comments
# Also, feel free to change the
# eigenvalue-modelling module to ResNet or other designs
# but whether this is a good choice is entirelly up to you ~(0 w <) ~
class MyMLP(nn.Module):

    def __init__(self, OD):
        super(MyMLP, self).__init__()
        
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, OD)

        self.init_weight()

    def init_weight(self):
        
        for module in self.parameters():
            if isinstance(module, nn.Linear):
                module.weight.data.copy_(
                    module.weight.data * 0.01)
                module.bias.data.copy_(
                    module.bias.data * 0.01)

    def forward(self, inputs, UpdateTrain = False):

        x = inputs.view(-1, 28 * 28)
        if UpdateTrain:
            x = Fnc.dropout(x, p = 0.1)
            
        x = torch.relu(self.linear1(x))
        if UpdateTrain:
            x = Fnc.dropout(x, p = 0.25)
        
        x = self.linear2(x)
        if UpdateTrain:
            x = Fnc.dropout(x, p = 0.1)

        return x

###===###
# the usual pre-processing of the gradient...
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
# Ok, time to comment further
# Apologies for the non-clean coding
class RNNOptimiser(nn.Module):

    def __init__(self, model, ID, HD, BS):
        super(RNNOptimiser, self).__init__()
        self.RefM = model

        self.ID = ID
        self.HD = HD

        self.ih_L1 = nn.Linear(2, HD, bias = False)

        stdv1 = 1.0 / math.sqrt(HD) 
        nn.init.uniform_(self.ih_L1.weight, -stdv1, stdv1) 

        ####===###
        # Ok, so in the B001d file we have
        #   self.hh_L1 = nn.Linear(HD, 4 * HD)
        # as the forward synaptic connections
        # and now in order to SVD for
        # all the gates seperately...
        # we ended up with the huge chunk below....

        ###===###
        # First we define the eigenvalue modelling procedure
        self.MyMLP1_x = MyMLP(HD)

        # Then for each forward synapse,
        # we SVD it
        # and create the fixed orthorgonal weights
        # of S and D
        # For instance,
        # F1x_S should be read as
        #   the S orthogonal weight of
        #   the x-related synapse
        #   in layer 1
        #   for the Forget gate ....
        self.F1x_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.F1x_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.I1x_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.I1x_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.A1x_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.A1x_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.O1x_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.O1x_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()
        
        ###===###
        # and now we do the same for the recurrent synapses...
        
        self.MyMLP1_q = MyMLP(HD)

        self.F1q_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.F1q_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.I1q_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.I1q_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.A1q_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.A1q_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.O1q_S  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).float().cuda()
        self.O1q_D  = torch.tensor(
                        ortho_group.rvs(dim = HD)
                        ).t().float().cuda()

        self.F1_bias = nn.Parameter(torch.ones(self.HD))
        self.I1_bias = nn.Parameter(-torch.ones(self.HD))

        ###===###
        # Such is much simpler for the second layer
        # as the dimension for each gated unit in L2 is 1
        # see B001d for reasons...
        self.MyMLP2_x = MyMLP(4)

        self.SchF2x = nn.Linear(1, self.HD, bias = False)
        self.SchI2x = nn.Linear(1, self.HD, bias = False)
        self.SchA2x = nn.Linear(1, self.HD, bias = False)
        self.SchO2x = nn.Linear(1, self.HD, bias = False)

        stdv2 = 1.0 / math.sqrt(1) 
        nn.init.uniform_(self.SchF2x.weight, -stdv2, stdv2)
        nn.init.uniform_(self.SchI2x.weight, -stdv2, stdv2)
        nn.init.uniform_(self.SchA2x.weight, -stdv2, stdv2)
        nn.init.uniform_(self.SchO2x.weight, -stdv2, stdv2)        
        
        self.MyMLP2_q = MyMLP(4)

        self.F2_bias = nn.Parameter(torch.ones(1))
        self.I2_bias = nn.Parameter(-torch.ones(1))

    ###===###
    # ROH is mainly the same as its B001d counterpart...skip...
    def ROH(self, keep_states=False, model=None):

        num_layers = 2
        #---
        self.RefM.reset()
        self.RefM.copy_params_from(model)

        if keep_states:
            for i in range(num_layers):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)

            self.E1 = Variable(self.E1.data)
            self.E2 = Variable(self.E2.data)
            self.E3 = Variable(self.E3.data)
            self.E4 = Variable(self.E4.data)
            
        else:
            self.hx = []
            self.cx = []
            self.hx.append(Variable(torch.zeros(1, self.HD)))
            self.cx.append(Variable(torch.zeros(1, self.HD)))
            self.hx.append(Variable(torch.zeros(1, 1)))
            self.cx.append(Variable(torch.zeros(1, 1)))            
            for i in range(num_layers):
                self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

            self.E1 = Variable(torch.zeros(self.HD)).cuda()
            self.E2 = Variable(torch.zeros(self.HD)).cuda()
            self.E3 = Variable(torch.zeros(4)).cuda()
            self.E4 = Variable(torch.zeros(4)).cuda()

    def forward(self, x, imgs, UpdateTrain = False):
          
        #---
        self.hx[0] = self.hx[0].expand(x.size(0), self.hx[0].size(1))
        self.cx[0] = self.cx[0].expand(x.size(0), self.cx[0].size(1))
        Q_1     = self.hx[0]
        S_1     = self.cx[0]

        ###===###
        # Ok, here is where we get to work
        # the N(x_t) part of Equation (15) on page 4
        # feature extraction the MLP for modelling the eigenvalues
        Img_features_x     = self.MyMLP1_x(imgs, UpdateTrain).transpose(0, 1)
        Img_statistics_x   = torch.mean(Img_features_x, dim = 1)
        Img_statistics_x   = torch.relu(Img_statistics_x)

        # The following is explained in Section B.5.3 on page 13
        # it is essentially Equation (17) and
        # is for increasing the stability of the network
        # feel free to use a non-adaptive version of this
        #   i.e. change 0.9 -> 0; and 0.1 -> 1
        Img_statistics_x   = 0.9 * self.E1 + 0.1 * Img_statistics_x
        self.E1 = Img_statistics_x

        # and this diagonalisation is the
        # diag() part of Equation (15) on page 4
        Img_EigenValues_x  = torch.diag(Img_statistics_x)

        # now we prepare for the variable-weights of MTL2L
        # LSTM1_FX should be read as
        #   the adaptive x-related weights
        #   for the forget gate
        #   of layer 1
        #   of the LSTM-like network
        LSTM1_FX        = torch.matmul(
                            torch.matmul(self.F1x_S, Img_EigenValues_x),
                            self.F1x_D)
        LSTM1_IX        = torch.matmul(
                            torch.matmul(self.I1x_S, Img_EigenValues_x),
                            self.I1x_D)
        LSTM1_AX        = torch.matmul(
                            torch.matmul(self.A1x_S, Img_EigenValues_x),
                            self.A1x_D)
        LSTM1_OX        = torch.matmul(
                            torch.matmul(self.O1x_S, Img_EigenValues_x),
                            self.O1x_D)

        # after preparing for all the synapses
        # let us find the gated values
        preact  = self.ih_L1(x)

        F = torch.matmul(preact, LSTM1_FX.transpose(0, 1))
        I = torch.matmul(preact, LSTM1_IX.transpose(0, 1))
        A = torch.matmul(preact, LSTM1_AX.transpose(0, 1))
        O = torch.matmul(preact, LSTM1_OX.transpose(0, 1))
                
        ###===###
        # and repeat this process for the recurrent synpatic connections
        Img_features_q     = self.MyMLP1_q(imgs, UpdateTrain).transpose(0, 1)
        Img_statistics_q   = torch.mean(Img_features_q, dim = 1)
        Img_statistics_q   = torch.relu(Img_statistics_q)

        Img_statistics_q   = 0.9 * self.E2 + 0.1 * Img_statistics_q
        self.E2 = Img_statistics_q
        
        Img_EigenValues_q  = torch.diag(Img_statistics_q)

        LSTM1_FQ        = torch.matmul(
                            torch.matmul(self.F1q_S, Img_EigenValues_q),
                            self.F1q_D)
        LSTM1_IQ        = torch.matmul(
                            torch.matmul(self.I1q_S, Img_EigenValues_q),
                            self.I1q_D)
        LSTM1_AQ        = torch.matmul(
                            torch.matmul(self.A1q_S, Img_EigenValues_q),
                            self.A1q_D)
        LSTM1_OQ        = torch.matmul(
                            torch.matmul(self.O1q_S, Img_EigenValues_q),
                            self.O1q_D)

        #---
        Fq_1 = Fnc.linear(Q_1,     LSTM1_FQ)
        Iq_1 = Fnc.linear(Q_1,     LSTM1_IQ)
        Aq_1 = Fnc.linear(Q_1,     LSTM1_AQ)
        Oq_1 = Fnc.linear(Q_1,     LSTM1_OQ)
        
        ###===###
        # gone through all the work to piece together
        # the forward bit and the recurrent bit
        F_1 = torch.sigmoid(F + Fq_1 + self.F1_bias)
        I_1 = torch.sigmoid(I + Iq_1 + self.I1_bias)
        A_1 = torch.tanh(   A + Aq_1)
        O_1 = torch.sigmoid(O + Oq_1)

        ###===###
        # and for updating the hidden variables
        S_1 = F_1 * S_1 +I_1 * A_1
        Q_1 = O_1 * torch.tanh(S_1)

        self.hx[0] = Q_1
        self.cx[0] = S_1
        
        ###===###
        # Repeat again for the 2nd layer
        x = self.hx[0]
        self.hx[1] = self.hx[1].expand(x.size(0), self.hx[1].size(1))
        self.cx[1] = self.cx[1].expand(x.size(0), self.cx[1].size(1))
        Q_2     = self.hx[1]
        S_2     = self.cx[1]

        Img_features_x     = self.MyMLP2_x(imgs, UpdateTrain).transpose(0, 1)
        Img_EigenValues_x   = torch.mean(Img_features_x, dim = 1)

        Img_EigenValues_x   = 0.9 * self.E3 + 0.1 * Img_EigenValues_x
        self.E3 = Img_EigenValues_x

        LSTM2_FX, LSTM2_IX, LSTM2_AX, LSTM2_OX = \
                  Img_EigenValues_x.unsqueeze(0).chunk(4, dim = 1)

        LSTM2_FX = self.SchF2x(LSTM2_FX).transpose(0, 1)
        LSTM2_IX = self.SchI2x(LSTM2_IX).transpose(0, 1)
        LSTM2_AX = self.SchA2x(LSTM2_AX).transpose(0, 1)
        LSTM2_OX = self.SchO2x(LSTM2_OX).transpose(0, 1)

        preact  = x

        F = torch.matmul(preact, LSTM2_FX)
        I = torch.matmul(preact, LSTM2_IX)
        A = torch.matmul(preact, LSTM2_AX)
        O = torch.matmul(preact, LSTM2_OX)
                
        #---
        Img_features_q     = self.MyMLP2_q(imgs, UpdateTrain).transpose(0, 1)
        Img_EigenValues_q  = torch.mean(Img_features_q, dim = 1)

        Img_EigenValues_q   = 0.9 * self.E4 + 0.1 * Img_EigenValues_q
        self.E4 = Img_EigenValues_q
        
        LSTM2_FQ, LSTM2_IQ, LSTM2_AQ, LSTM2_OQ = \
                  Img_EigenValues_q.unsqueeze(0).chunk(4, dim = 1)
        
        #---
        Fq_2 = Q_2 * LSTM2_FQ
        Iq_2 = Q_2 * LSTM2_IQ
        Aq_2 = Q_2 * LSTM2_AQ
        Oq_2 = Q_2 * LSTM2_OQ
        
        #---    
        F_2 = torch.sigmoid(F + Fq_2 + self.F2_bias)
        I_2 = torch.sigmoid(I + Iq_2 + self.I2_bias)
        A_2 = torch.tanh(   A + Aq_2)
        O_2 = torch.sigmoid(O + Oq_2)

        S_2 = F_2 * S_2 +I_2 * A_2
        Q_2 = O_2 * torch.tanh(S_2)

        self.hx[1] = Q_2
        self.cx[1] = S_2    
        
        #---
        Q_2 = Q_2.squeeze(1)

        ###===###
        # and done... here we have it
        # refer to why we * 0.1 in B001d
        return Q_2 * 0.1

    def UpdateTransfer(self, CurOptimisee, imgs, \
                       train = False, t = 0):

        grads = []
        
        for module in CurOptimisee.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.RefM.get_flat_params()
        flat_params = flat_params.unsqueeze(1)
        flat_grads = preprocess_gradients(torch.cat(grads))

        flat_params = flat_params.squeeze(1)

        #---
        Update = self(flat_grads, imgs, UpdateTrain = train)
        flat_params = flat_params + Update

        if train:
            if torch.randn(1) < 0:
                CFM = torch.max(abs(flat_params)).detach() * (100 - t) / 100 * 0.025
            else:
                CFM = torch.max(abs(flat_params)).detach() * (100 - t) / 100 * 0.05
            flat_params += CFM * torch.randn_like(flat_params)    

        self.RefM.set_flat_params(flat_params)

        self.RefM.copy_params_to(CurOptimisee)
        return self.RefM.model      

###===###
# ... and other stuff
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
