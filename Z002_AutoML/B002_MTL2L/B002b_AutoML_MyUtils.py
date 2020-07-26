###===###
# basically identical to its B001b counterpart

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

from    B002c_AutoML_BaseLearner        import  MyMLP

###===###
TestSteps   = 1000
TestPlot    = 10

DFLT = True

if DFLT:
    TTrial  = 10

else:
    TTrial  = 1

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

def testphasetest(Optimiser, train_loader, test_loader, unroll):

    allacc = []
    alllos = []
    C1000 = []
    L1000 = []
    C2000 = []
    L2000 = []
    C3000 = []
    L3000 = []
                
    for itr in range(TTrial):

        if itr+1 > 1:
            plt.subplot(222)
            plt.cla()
            plt.draw()
            plt.pause(1)
            plt.title("Testing Acc")

            #---
            plt.subplot(224)
            plt.cla()
            plt.draw()
            plt.pause(1)

            plt.title("Testing Loss")               
            

        print('this is testing simulation numero {}'.format(itr))
        TTstep200 = []
        TTloss200 = []
        TTcstepF  = []
        
        ccstep = 0
        
        model = MyMLP()
        model.cuda()

        for k in range(TestSteps // unroll):

            train_iter = iter(train_loader)

            Optimiser.ROH(keep_states=k > 0, model=model)

            loss_sum = 0
            prev_loss = torch.zeros(1)
            prev_loss = prev_loss.cuda()

            for j in range(unroll):

                ccstep += 1
                if np.mod(ccstep, int(TestSteps/TestPlot)) == 0:
                    print('testing simulation {}, now {}, all {}'.format(itr+1, ccstep, TestSteps))
                    test_loss, cur_acc = test(RefModel, test_loader)
                    TTstep200.append(cur_acc)
                    TTloss200.append(test_loss)
                    TTcstepF.append(ccstep)

                    plt.subplot(222)
                    plt.title("Testing Acc")                  
                    plt.ylim([-0.125, 1.125])
                    plt.xlim([-TestSteps/200*10,
                              TestSteps + TestSteps/200+10])
                    plt.plot(TTcstepF, TTstep200, '-o', color = 'blue')
                    plt.draw()

                    plt.pause(1)

                    plt.subplot(224)
                    plt.title("Testing Loss")
                    plt.ylim([0, 0.15])
                    plt.xlim([-TestSteps/200*10,
                              TestSteps + TestSteps/200+10])
                    plt.plot(TTcstepF, TTloss200, '-o', color = 'pink')
                    plt.draw()

                    plt.pause(1)                    

                if ccstep == TestSteps:
                    allacc.append(cur_acc)
                    alllos.append(test_loss)

                if ccstep == 1000:
                    C1000.append(cur_acc)
                    L1000.append(test_loss)
                if ccstep == 2000:
                    C2000.append(cur_acc)
                    L2000.append(test_loss)
                if ccstep == 3000:
                    C3000.append(cur_acc)
                    L3000.append(test_loss)
                            
                x, y = next(train_iter)              
                x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                f_x  = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward()
                RefModel = Optimiser.UpdateTransfer(model, x)

    ###===###
    allacc = np.array(allacc) * 100
    alllos = np.array(alllos)

    C1000 = np.array(C1000) * 100
    L1000 = np.array(L1000)
    C2000 = np.array(C2000) * 100
    L2000 = np.array(L2000)
    C3000 = np.array(C3000) * 100
    L3000 = np.array(L3000)
    
    return allacc, alllos, C3000, L3000, C2000, L2000, C1000, L1000
