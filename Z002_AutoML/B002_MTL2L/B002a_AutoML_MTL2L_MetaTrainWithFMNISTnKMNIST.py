###===###
# This is 1/2 the main script file of our paper
# Therein this script,
# we meta-train our novel MTL2L neural optimiser
# on both FashionMNIST and KMNIST as according to Figure 3
# on page 3 of our paper
# https://arxiv.org/pdf/2007.09343.pdf
#
#---
# We will not add redundant comments,
# as most of the code has already been explained in
# see B001a in details

###===###
LSize       = 25450
NO_HD       = 20

#---
# most of the meta-training related setup are the same
# besides LTrial, the Q-factor in our paper
# the amount of trials is increase to allow for
# sufficient rehersal between both FashionMNIST and KMNIST
TrainSteps  = 100
NO_Unroll   = 20
LTrial      = 50 # <- this one here

#---
BS          = 128
TrainPlot   = 4
PTFrequency = 1

#---
# Observations:
#   During meta-training,
#   the first found of training
#       i.e. trial no. 1 addressing FashionMNIST &
#            trial no. 2 addressing KMNIST
#   will behave nicely
#   However, the second round for training
#       i.e. trial no. 3 addressing FashionMNIST &
#            trial no. 4 addressing KMNIST
#   will see some pretty bad outcomes
#   this is perhaps due to the severe reconfiguration of MTL2L
#   (catastrophic forgetting)
#   after learning to update the MLP base learner on 2 different
#   input domains
#   The network usually finds a common ground on round no. 5
#   but will require a relatively lengthy process to stabilise
#
#---
# The rest of the code is not commented,
# but check out B002d_AutoML_MTL2L for more comments

###===###
import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim
from    torch.autograd          import  Variable
from    torchvision             import  datasets, transforms
import  numpy                   as      np
import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms
import  matplotlib.pyplot       as      plt

#---
# Self-use
from    B002b_AutoML_MyUtils        import  test, testphasetest
from    B002c_AutoML_BaseLearner    import  MyMLP
from    B002d_AutoML_MTL2L          import  RefMode, RNNOptimiser

###===###
seed = 1111           
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed_all( seed)

###===###
kwargs = {'pin_memory': True}
tr_S_loader1 = torch.utils.data.DataLoader(
                    datasets.KMNIST('./data_K', train = True, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])),
                    batch_size = BS, shuffle = True, **kwargs)

tr_S_loader2 = torch.utils.data.DataLoader(
                    datasets.FashionMNIST('./data_F', train = True, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])),
                    batch_size = BS, shuffle = True, **kwargs)

te_S_loader1 = torch.utils.data.DataLoader(
                    datasets.KMNIST('./data_K', train = False, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])),
                    batch_size = BS, shuffle = True, **kwargs)

te_S_loader2 = torch.utils.data.DataLoader(
                    datasets.FashionMNIST('./data_F', train = False, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])),
                    batch_size = BS, shuffle = True, **kwargs) 

###===###
RefModel    = MyMLP()
RefModel    = RefModel.cuda()

Optimiser   = RNNOptimiser(RefMode(RefModel),   
                           ID = LSize,          
                           HD = NO_HD,
                           BS = BS)         
Optimiser.cuda()

OOO = optim.Adam(Optimiser.parameters(), lr=1e-3)

ccstep1 = 0
ccstep2 = 0
last200step1 = None
last200loss1 = None
last200step2 = None
last200loss2 = None
        
for i in range(LTrial):

    print('this is LTrial numero {}'.format(i))
    
    if np.mod(i + 1, PTFrequency) == 0:
        if np.mod(i+1, 2) == 0:
            if last200step1 is not None:
                plt.subplot(411)
                plt.title("KMNIST Training Acc")
                plt.cla()
                plt.draw()
                plt.pause(1)
                
                plt.subplot(411)
                plt.ylim([-0.125, 1.125])
                plt.xlim([-TrainSteps/200*10,
                          TrainSteps + TrainSteps/200+10])
                plt.plot(cstepF_1, last200step1, '-o', color = 'green')
                plt.draw()

                plt.pause(1)

                #---
                plt.subplot(412)
                plt.title("KMNIST Training Loss")
                plt.cla()
                plt.draw()
                plt.pause(1)
                
                plt.subplot(412)
                plt.ylim([0, 0.15])
                plt.xlim([-TrainSteps/200*10,
                          TrainSteps + TrainSteps/200+10])
                plt.plot(cstepF_1, last200loss1, '-o', color = 'red')
                plt.draw()

                plt.pause(1)            

            step200_1 = []
            loss200_1 = []
            cstepF_1  = []
        else:
            if last200step2 is not None:
                plt.subplot(413)
                plt.title("FashionMNIST Training Acc")
                plt.cla()
                plt.draw()
                plt.pause(1)
                
                plt.subplot(413)
                plt.ylim([-0.125, 1.125])
                plt.xlim([-TrainSteps/200*10,
                          TrainSteps + TrainSteps/200+10])
                plt.plot(cstepF_2, last200step2, '-o', color = 'green')
                plt.draw()

                plt.pause(1)

                #---
                plt.subplot(414)
                plt.title("FashionMNIST Training Loss")
                plt.cla()
                plt.draw()
                plt.pause(1)
                
                plt.subplot(414)
                plt.ylim([0, 0.15])
                plt.xlim([-TrainSteps/200*10,
                          TrainSteps + TrainSteps/200+10])
                plt.plot(cstepF_2, last200loss2, '-o', color = 'red')
                plt.draw()

                plt.pause(1)            

            step200_2 = []
            loss200_2 = []
            cstepF_2  = []
            
        
    ccstep1 = 0
    ccstep2 = 0
    
    model = MyMLP()
    model.cuda()

    for k in range(TrainSteps // NO_Unroll):

        if np.mod(i+1, 2) == 0:
            train_iter = iter(tr_S_loader1)
        else:
            train_iter = iter(tr_S_loader2)

        Optimiser.ROH(keep_states=k > 0, model=model)

        loss_sum = 0
        prev_loss = torch.zeros(1)
        prev_loss = prev_loss.cuda()

        for j in range(NO_Unroll):

            if np.mod(i+1, 2) == 0:
                ccstep1 += 1
                if np.mod(ccstep1, int(TrainSteps/TrainPlot)) == 0:
                    print('LTrial {}, now {}, all {}'.format(i+1, ccstep1, TrainSteps))
                    if np.mod(i+1, PTFrequency) == 0:

                        te_S_loader = iter(te_S_loader1)
     
                        test_loss, cur_acc = test(RefModel, te_S_loader)
                        step200_1.append(cur_acc)
                        loss200_1.append(test_loss)
                        cstepF_1.append(ccstep1)

                        plt.subplot(411)
                        plt.title("KMNIST Training Acc")
                        plt.ylim([-0.125, 1.125])
                        plt.xlim([-TrainSteps/200*10,
                                  TrainSteps + TrainSteps/200+10])
                        plt.plot(cstepF_1, step200_1, '-o', color = 'blue')
                        plt.draw()

                        plt.pause(1)

                        #---
                        plt.subplot(412)
                        plt.title("KMNIST Training Loss")
                        plt.ylim([0, 0.15])
                        plt.xlim([-TrainSteps/200*10,
                                  TrainSteps + TrainSteps/200+10])
                        plt.plot(cstepF_1, loss200_1, '-o', color = 'pink')
                        plt.draw()

                        plt.pause(1)

            else:
                ccstep2 += 1
                if np.mod(ccstep2, int(TrainSteps/TrainPlot)) == 0:
                    print('LTrial {}, now {}, all {}'.format(i+1, ccstep2, TrainSteps))
                    if np.mod(i+1, PTFrequency) == 0:

                        te_S_loader = iter(te_S_loader2)
                                
                        test_loss, cur_acc = test(RefModel, te_S_loader)
                        step200_2.append(cur_acc)
                        loss200_2.append(test_loss)
                        cstepF_2.append(ccstep2)

                        plt.subplot(413)
                        plt.title("FashionMNIST Training Acc")
                        plt.ylim([-0.125, 1.125])
                        plt.xlim([-TrainSteps/200*10,
                                  TrainSteps + TrainSteps/200+10])
                        plt.plot(cstepF_2, step200_2, '-o', color = 'blue')
                        plt.draw()

                        plt.pause(1)

                        #---
                        plt.subplot(414)
                        plt.title("FashionMNIST Training Loss")
                        plt.ylim([0, 0.15])
                        plt.xlim([-TrainSteps/200*10,
                                  TrainSteps + TrainSteps/200+10])
                        plt.plot(cstepF_2, loss200_2, '-o', color = 'pink')
                        plt.draw()

                        plt.pause(1)
                        
            x, y = next(train_iter)              
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            f_x  = model(x)
            loss = F.nll_loss(f_x, y)
            model.zero_grad()
            loss.backward()
            RefModel = Optimiser.UpdateTransfer(model, x, train = True, t = ccstep1) # put x in there
            f_x = RefModel(x)
            loss = F.nll_loss(f_x, y)
            loss_sum += (loss - Variable(prev_loss))
            prev_loss = loss.data

        Optimiser.zero_grad()
        loss_sum.backward()
        for param in Optimiser.parameters():
            try:
                param.grad.data.clamp_(-1, 1)
            except:
                continue
        OOO.step()

    ###===###       

    ###===###
    if np.mod(i+1, PTFrequency) == 0:
        if np.mod(i+1, 2) == 0:
            last200step1 = step200_1
            last200loss1 = loss200_1
        else:
            last200step2 = step200_2
            last200loss2 = loss200_2
        


