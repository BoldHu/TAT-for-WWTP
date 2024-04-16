import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import *
from utilities import *
from data import *
from networks import *
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

# log = Logger('log/sentiment-msda', clear=True)
cls = CLS(15, 1, bottle_neck_dim = 128).cuda()
discriminator = LargeDiscriminator(15).cuda()
scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
optimizer_cls = OptimWithSheduler(optim.Adam(cls.parameters(), weight_decay = 5e-4, lr = 5e-5),
                                  scheduler)
optimizer_discriminator = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay = 5e-4, lr = 5e-5),
                                  scheduler)


source_train_data = sio.loadmat('./Data/X_kla120.mat')['data']
target_train_data = sio.loadmat('./Data/X_kla240.mat')['data']
source_train_label = sio.loadmat('./Data/EQvec_kla120.mat')['EQvec']
target_train_label = sio.loadmat('./Data/EQvec_kla240.mat')['EQvec']

# standardlize the data
scaler = StandardScaler()
source_train_data = scaler.fit_transform(source_train_data)
target_train_data = scaler.fit_transform(target_train_data)

# =====================train
k=0
while k < 5:
    mini_batches_source = get_mini_batches(source_train_data, source_train_label, 64)
    mini_batches_target = get_mini_batches(target_train_data, target_train_label, 64)
    for (i, ((source_feature, source_label,), (target_feature, target_label,))) in enumerate(
            zip(mini_batches_source, mini_batches_target)):
        
        # =========================generate transferable examples
        source_label_0 = torch.from_numpy(source_label).cuda()
        feature_fooling = torch.from_numpy(target_feature).float().cuda()
        feature_fooling.requires_grad = True
        feature_fooling_c = torch.from_numpy(source_feature).float().cuda()
        feature_fooling_c.requires_grad = True
        feature_fooling_0 = feature_fooling.detach()
        
        for i in range(2):
            scores = discriminator(feature_fooling)
            loss = BCELossForMultiClassification(torch.ones_like(scores) , 1 - scores) - 0.1 * torch.sum((feature_fooling - feature_fooling_0) * (feature_fooling - feature_fooling_0))
            loss.backward()
            g = feature_fooling.grad
            feature_fooling = feature_fooling + 0.001 * g 
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling = feature_fooling.data.cpu().cuda()
            feature_fooling.requires_grad = True
        
        feature_fooling_c1 = feature_fooling_c.detach()
        for xs in range(2):
            scorec = discriminator.forward(feature_fooling_c)
            losss = BCELossForMultiClassification(torch.ones_like(scorec) ,  scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            losss.backward()
            gss = feature_fooling_c.grad
            feature_fooling_c = feature_fooling_c +  0.001 * gss
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling_c = feature_fooling_c.data.cpu().cuda()
            feature_fooling_c.requires_grad = True
        
        for xss in range(3):
            # _,_,_,scorec = cls.forward(feature_fooling_c)
            _,_,scorec = cls.forward(feature_fooling_c)
            # loss = CrossEntropyLoss(source_label_0, scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            loss = MSELoss(source_label_0, scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            loss.backward()
            gs = feature_fooling_c.grad
            feature_fooling_c = feature_fooling_c +  0.001 * gs
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling_c = feature_fooling_c.data.cpu().cuda()
            feature_fooling_c.requires_grad = True
            
        # =========================forward pass
        feature_source = torch.from_numpy(source_feature).float().cuda()
        source_label = torch.from_numpy(source_label).float().cuda()
        feature_target = torch.from_numpy(target_feature).float().cuda()
        target_label = torch.from_numpy(target_label).float().cuda()
        
        # _, _, __, predict_prob_source = cls.forward(feature_source)
        # _, _, __, predict_prob_target = cls.forward(feature_target)
        # _, _, __, predict_prob_fooling = cls.forward(feature_fooling)
        # _, _, __, predict_prob_fooling_c = cls.forward(feature_fooling_c)
        _, _, predict_prob_source = cls.forward(feature_source)
        _, _, predict_prob_target = cls.forward(feature_target)
        _, _, predict_prob_fooling = cls.forward(feature_fooling)
        _, _, predict_prob_fooling_c = cls.forward(feature_fooling_c)

        domain_prob_source = discriminator.forward(feature_source)
        domain_prob_target = discriminator.forward(feature_target)
        domain_prob_fooling = discriminator.forward(feature_fooling)
        domain_prob_fooling_c = discriminator.forward(feature_fooling_c)
        
        dloss = BCELossForMultiClassification(torch.ones_like(domain_prob_source), domain_prob_source)
        dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_target), 1 - domain_prob_target)
        dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_fooling_c), domain_prob_fooling_c.detach())
        dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_fooling), 1 - domain_prob_fooling.detach())
        
        # ce = CrossEntropyLoss(source_label, predict_prob_source)
        # ce_extra_c = CrossEntropyLoss(source_label, predict_prob_fooling_c)
        ce = MSELoss(source_label, predict_prob_source)
        ce_extra_c = MSELoss(source_label, predict_prob_fooling_c)
        

        dis = torch.sum((predict_prob_fooling - predict_prob_target) * (predict_prob_fooling - predict_prob_target))
        

        with OptimizerManager([optimizer_cls , optimizer_discriminator]):
            loss = ce + ce_extra_c + dis + dloss
            loss.backward()
                        
        # log.step += 1
    
        # if log.step % 10 == 1:
        #     counter = AccuracyCounter()
        #     counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(source_label))
        #     acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype = np.float32))).cuda()
        #     track_scalars(log, ['ce', 'acc_train', 'dis','ce_extra_c','dloss'], globals())

        # if log.step % 100 == 0:
        #     clear_output()
    k += 1
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ======================test
with TrainingModeManager([cls], train = False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
    for (i, (feature, label)) in enumerate(mini_batches_target):
        # fs = Variable(torch.from_numpy(feature), volatile = True).cuda()
        # fs = torch.from_numpy(feature).to(device)
        # label = torch.from_numpy(label).to(device)
        # label = Variable(torch.from_numpy(label), volatile = True).cuda()
        fs = torch.from_numpy(feature).float().to(device)
        label = torch.from_numpy(label).float().to(device)
        # __, fs, _, predict_prob = cls.forward(fs)
        _, fs, predict_prob = cls.forward(fs)
        predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
        label = np.argmax(label, axis = -1).reshape(-1, 1)
        predict_index = np.argmax(predict_prob, axis = -1).reshape(-1, 1)
        accumulator.updateData(globals())
        if i % 10 == 0:
            print(i)

for x in accumulator.keys():
    globals()[x] = accumulator[x]
print('acc')
print(float(np.sum(label.flatten() == predict_index.flatten()) )/ label.flatten().shape[0])


