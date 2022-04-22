from HRN import Net
from thop import profile
import torch
import sys
#f = open('model.log', 'a')
#sys.stdout = f
input = torch.randn(1, 3, 32, 32)
model = Net(scale=4,group=1)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    param = []
    names = []
    i =j=1 
    for p in net.parameters():
        print(i,p.numel())
        i=i+1
        param.append(p.numel())
    for name in net.state_dict():
        print(j,name)
        j =j+1
        names.append(name)
    print(len(param),len(names))
    with open('flops_params.txt','a') as f:
        for i in range(len(param)):
            f.write(names[i]+":"+str(param[i])+"\n")
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

dic =  get_parameter_number(model)
print(dic)


flops, params = profile(model, inputs=(input, ))
print(flops)
from thop import clever_format

macs, params = clever_format([flops, params], "%.3f")
print('macs:',macs)
print('params:',params)
