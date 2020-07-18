##==##
# Dependencies
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


###===###
if 1:
    PLHDR = torch.load('A001b_AutoML_Data.MNIST_SGD')
    PLHDR = torch.cat(PLHDR, dim = 1)
    PLHDR = torch.mean(PLHDR, dim = 1).numpy()
    LE10_SGD = PLHDR
    
#---
if 1:
    PLHDR = torch.load('A001c_AutoML_Data.MNIST_SGDm')
    PLHDR = torch.cat(PLHDR, dim = 1)
    PLHDR = torch.mean(PLHDR, dim = 1).numpy()
    LE10_SGDm = PLHDR

#---
if 1:
    PLHDR = torch.load('A001d_AutoML_Data.MNIST_Adam')
    PLHDR = torch.cat(PLHDR, dim = 1)
    PLHDR = torch.mean(PLHDR, dim = 1).numpy()
    LE10_Adam = PLHDR

#---
if 1:
    PLHDR = torch.load('A001e_AutoML_Data.MNIST_LSTMoptimiser')
    PLHDR = [torch.tensor(i).unsqueeze(1) for i in PLHDR]
    PLHDR = torch.cat(PLHDR, dim = 1)
    PLHDR = torch.mean(PLHDR, dim = 1).numpy()
    LE10_LSTM = PLHDR

#---
if 1:
    PLHDR = torch.load('A003f_AutoML_Data.MNIST_MTL2L')
    PLHDR = [torch.tensor(i).unsqueeze(1) for i in PLHDR]
    PLHDR = torch.cat(PLHDR, dim = 1)
    PLHDR = torch.mean(PLHDR, dim = 1).numpy()
    LE10_MTL2L = PLHDR

###===###
if 1:
    Head_NO = max(LE10_SGD[0], LE10_SGDm[0], LE10_Adam[0], LE10_LSTM[0])

    Loss_SGD = [Head_NO]
    Loss_SGD.extend(LE10_SGD)
    Loss_SGDm = [Head_NO]
    Loss_SGDm.extend(LE10_SGDm)
    Loss_Adam = [Head_NO]
    Loss_Adam.extend(LE10_Adam)
    Loss_LSTM = [Head_NO]
    Loss_LSTM.extend(LE10_LSTM)
    Loss_MTL2L = [Head_NO]
    Loss_MTL2L.extend(LE10_MTL2L)

#---
x_axis = [i for i in range(0, 1010, 10)]

#---
plt.plot(x_axis, Loss_LSTM,
         linewidth = 3, linestyle = '-',    color = 'cyan')
plt.plot(x_axis, Loss_SGD,
         linestyle = '', color = 'red',     marker = 'x')
plt.plot(x_axis, Loss_SGDm,
         linestyle = '', color = 'orange',  marker = '+')
plt.plot(x_axis, Loss_Adam,
         linestyle = '', color = 'blue',    marker = '*')
plt.plot(x_axis, Loss_MTL2L,
             linewidth = 3, linestyle = '-',    color = 'black')

#---
plt.legend(('L-optimiser', 'SGD', 'SGD w/ momentum', 'ADAM', 'MTL2L'),
           fontsize=15)
plt.xticks(fontsize=15, rotation=0)
plt.yticks(fontsize=15, rotation=0)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Loss', fontsize=15, rotation = 90)
plt.title('Meta-Trained on MNIST; Meta-tested on MNIST', fontsize=15)
plt.show()



