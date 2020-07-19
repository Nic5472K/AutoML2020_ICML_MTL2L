###===###
# Hyperparameters
#---
# amount of parameters in the learner
LSize       = 25450
# the hidden size of the nueral optimiser
NO_HD       = 20
# train the neural optimiser with 100 steps
TrainSteps  = 100
# and unroll every 20 steps
NO_Unroll   = 20
# the neural optimiser is trained to update 10 randomly initialised learner
LTrial      = 10 # 30
# batch size
BS          = 128
# misc
TrainPlot   = 4
PTFrequency = 1
    
###===###
# Dependencies
#--
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

#--
# test function
from    B001b_AutoML_MyUtils        import  test
# architectural definition of the base learner
from    B001c_AutoML_BaseLearner    import  MyMLP
# definition of the LSTMoptimiser
from    B001d_AutoML_LSTMoptimiser  import  RefMode, RNNOptimiser

###===###
seed = 101           
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed_all( seed)

###===###
# Setting up datasets
kwargs = {'pin_memory': True}
tr_S_loader1 = torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train = True, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
                    batch_size = BS, shuffle = True, **kwargs)

te_S_loader1 = torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train = False, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
                    batch_size = BS, shuffle = True, **kwargs)


###===###
# set up the base learner
RefModel    = MyMLP()
RefModel    = RefModel.cuda()

# setup the RNN optimiser
Optimiser   = RNNOptimiser(RefMode(RefModel),   
                           ID = LSize,          
                           HD = NO_HD
                           )         
Optimiser.cuda()

# setup the optimiser used to update the neural optimiser
OOO = optim.Adam(Optimiser.parameters(), lr=1e-3)

# misc
ccstep1 = 0
last200step1 = None
last200loss1 = None

###===###
# start the heavy lifting
for i in range(LTrial):

    print('this is LTrial numero {}'.format(i))

    #---
    # plotting related
    if np.mod(i + 1, PTFrequency) == 0:
        # plot the progress
        if last200step1 is not None:
            plt.subplot(211)
            plt.title("MNIST Training Acc")
            plt.cla()
            plt.draw()
            plt.pause(1)
            
            plt.subplot(211)
            plt.ylim([-0.125, 1.125])
            plt.xlim([-TrainSteps/200*10,
                      TrainSteps + TrainSteps/200+10])
            plt.plot(cstepF_1, last200step1, '-o', color = 'green')
            plt.draw()

            plt.pause(1)

            #---
            plt.subplot(212)
            plt.title("MNIST Training Loss")
            plt.cla()
            plt.draw()
            plt.pause(1)
            
            plt.subplot(212)
            plt.ylim([0, 0.15])
            plt.xlim([-TrainSteps/200*10,
                      TrainSteps + TrainSteps/200+10])
            plt.plot(cstepF_1, last200loss1, '-o', color = 'red')
            plt.draw()

            plt.pause(1)            

        step200_1 = []
        loss200_1 = []
        cstepF_1  = []
                    
    ccstep1 = 0

    #---
    # for every trial, randomly initialise a learner network
    model = MyMLP()
    model.cuda()

    #---
    # for every clusters of unrolling steps 
    for k in range(TrainSteps // NO_Unroll):

        # get new data
        train_iter = iter(tr_S_loader1)

        # reset optimiser hidden state
        Optimiser.ROH(keep_states=k > 0, model=model)

        # redefined the cumulative loss for updating the neural optimiser
        loss_sum = 0
        prev_loss = torch.zeros(1)
        prev_loss = prev_loss.cuda()

        #---
        # for all steps before an unroll occurs
        for j in range(NO_Unroll):

            #---
            # plot related
            ccstep1 += 1
            if np.mod(ccstep1, int(TrainSteps/TrainPlot)) == 0:
                print('LTrial {}, now {}, all {}'.format(i+1, ccstep1, TrainSteps))
                if np.mod(i+1, PTFrequency) == 0:

                    te_S_loader = iter(te_S_loader1)
 
                    test_loss, cur_acc = test(RefModel, te_S_loader)
                    step200_1.append(cur_acc)
                    loss200_1.append(test_loss)
                    cstepF_1.append(ccstep1)

                    plt.subplot(211)
                    plt.title("MNIST Training Acc")
                    plt.ylim([-0.125, 1.125])
                    plt.xlim([-TrainSteps/200*10,
                              TrainSteps + TrainSteps/200+10])
                    plt.plot(cstepF_1, step200_1, '-o', color = 'blue')
                    plt.draw()

                    plt.pause(1)

                    #---
                    plt.subplot(212)
                    plt.title("MNIST Training Loss")
                    plt.ylim([0, 0.15])
                    plt.xlim([-TrainSteps/200*10,
                              TrainSteps + TrainSteps/200+10])
                    plt.plot(cstepF_1, loss200_1, '-o', color = 'pink')
                    plt.draw()

                    plt.pause(1)

            #---
            # get a new pair of inputs + labels
            x, y = next(train_iter)              
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            # feed it to the current base learner
            f_x  = model(x)
            # find the gradient of the loss of this base learner
            loss = F.nll_loss(f_x, y)
            model.zero_grad()
            loss.backward()
            # forward it towards the neural optimiser
            # and update the current base learner
            RefModel = Optimiser.UpdateTransfer(model, x)
            f_x = RefModel(x)
            # find the loss after the update
            loss = F.nll_loss(f_x, y)
            # the cumulative information of
            # the optentiality of improving the base learner
            # is used for updating the base learner
            loss_sum += (loss - Variable(prev_loss))
            prev_loss = loss.data

        # after all steps before unrolling the neural optimiser
        # update the neural optimiser
        Optimiser.zero_grad()
        loss_sum.backward()
        for param in Optimiser.parameters():
            try:
                param.grad.data.clamp_(-1, 1)
            except:
                continue
        OOO.step()

    #---
    # plotting related
    if np.mod(i+1, PTFrequency) == 0:
        last200step1 = step200_1
        last200loss1 = loss200_1
        


