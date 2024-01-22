'''
To get the pretrained model, you need to:

pip install git+https://github.com/RobustBench/robustbench.git

For other instructions, please visit: https://github.com/RobustBench/robustbench
'''



# For Cifar-10/Cifar-10-C/Cifar-Flip: WideResNet-28-10
from robustbench.utils import load_model

model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
print(model)


# For Imagenet/Imagenet-C/Living-17/Entity-30/Waterbird: ResNet-50
model = load_model(model_name='Standard_R50', dataset='imagenet', threat_model='Linf')

print(model)
