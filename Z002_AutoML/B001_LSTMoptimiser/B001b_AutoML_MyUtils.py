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
from    B001c_AutoML_BaseLearner        import  MyMLP

###===###
TestSteps   = 1000
TestPlot    = 10
DFLT        = True
TTrial      = 10

def test(model, test_loader):
    test_loss = 0
    correct = 0
    tot_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.cuda(), target.cuda()

            output     = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred       = output.argmax(dim=1, keepdim=True)
            correct   += pred.eq(target.view_as(pred)).sum().item()
            tot_num   += len(target)

    test_loss /= tot_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, tot_num,
        100. * correct / tot_num))

    return test_loss, correct / tot_num
