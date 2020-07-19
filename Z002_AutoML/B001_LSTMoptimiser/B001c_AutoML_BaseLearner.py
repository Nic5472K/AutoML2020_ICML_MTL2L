import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim
from    torch.autograd          import  Variable


class MyMLP(nn.Module):

    def __init__(self):
        super(MyMLP, self).__init__()
        
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

        self.init_weight()

    def init_weight(self):
        
        for module in self.parameters():
            if isinstance(module, nn.Linear):
                module.weight.data.copy_(
                    module.weight.data * 0.01)
                module.bias.data.copy_(
                    module.bias.data * 0.01)

    def forward(self, inputs):

        x = inputs.view(-1, 28 * 28)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.log_softmax(x, dim = 1)

        return x
