# calculat the sum parameter of three models
import torch
from thop import profile

cls = torch.load('models/cls_model_X_kla240.mat_X_mu0.7.mat.pt')
dis = torch.load('models/discriminator_model_X_kla240.mat_X_mu0.7.mat.pt')
fea_ext = torch.load('models/feature_extractor_model_X_kla240.mat_X_mu0.7.mat.pt')
cls = cls.to('cuda')
dis = dis.to('cuda')
fea_ext = fea_ext.to('cuda')

input_fea = torch.randn(32,15)
input_fea = input_fea.to('cuda')
input_cls = torch.randn(32,128)
input_cls = input_cls.to('cuda')
input_dis = torch.randn(32,128)
input_dis = input_dis.to('cuda')
flops_cls, params_cls = profile(cls, inputs=(input_cls,))
flops_dis, params_dis = profile(dis, inputs=(input_dis,))
flops_fea_ext, params_fea_ext = profile(fea_ext, inputs=(input_fea,))
print('cls flops: ', flops_cls)
print('cls params: ', params_cls)
print('dis flops: ', flops_dis)
print('dis params: ', params_dis)
print('fea_ext flops: ', flops_fea_ext)
print('fea_ext params: ', params_fea_ext)
print('total flops: ', flops_cls+flops_dis+flops_fea_ext)
print('total params: ', params_cls+params_dis+params_fea_ext)


