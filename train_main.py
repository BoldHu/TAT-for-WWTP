import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import *
from utilities import inverseDecaySheduler, OptimWithSheduler, OptimizerManager, BCELossForMultiClassification, MSELoss
from data import *
from networks import *
from data_loader import GetLoader
from torch.utils.data import DataLoader
from torch.backends import cudnn
from reg_functions import reg_indicator
from remove_word import remove, change
from test import test
import random

def train(source_feature, source_label, target_feature, target_label):
    
    # hyperparameters setting
    batch_size = 32
    num_epochs = 200
    cuda = True
    cudnn.benchmark = True    
    feature_extractor = FeatureExtractor(in_dim=15, out_dim=128).cuda()
    cls = CLS(in_dim=128, out_dim=1, bottle_neck_dim=512).cuda()
    discriminator = LargeDiscriminator(in_feature=128).cuda()
    # set the random seed
    random.seed(42)
    torch.manual_seed(42)

    scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
    optimizer_cls = OptimWithSheduler(optim.Adam(cls.parameters(), weight_decay = 3e-4, lr = 3e-5),
                                    scheduler)
    optimizer_discriminator = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay = 3e-4, lr = 3e-5),
                                    scheduler)

    # =====================load data
    source_dataset_name = source_feature
    target_dataset_name = target_feature
    source_label_name = source_label
    target_label_name = target_label
    # change name to the test file name
    test_source_dataset_name = change(source_dataset_name)
    test_target_dataset_name = change(target_dataset_name)
    test_source_label_name = change(source_label_name)
    test_target_label_name = change(target_label_name)
    
    source_dataset = GetLoader(source_dataset_name, source_label_name, transform=True)
    target_dataset = GetLoader(target_dataset_name, target_label_name, transform=True)
    
    source_dataloader = DataLoader(dataset = source_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)
    target_dataloader = DataLoader(dataset = target_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)
    print('train start')
    print('source dataset size:{0}, target dataset size:{1}'.format(len(source_dataset), len(target_dataset)))

    # =====================train
    k=0
    source_r2_max = -1000000000000000
    target_r2_max = -1000000000000000
    source_rmse_min = 100000000000
    target_rmse_min = 100000000000
    while k < num_epochs:
        for (_, (source, target)) in enumerate(zip(source_dataloader, target_dataloader)):
            source_feature, source_label = source
            target_feature, target_label = target
            # featur extract
            source_feature = feature_extractor.forward(source_feature.float().cuda())
            target_feature = feature_extractor.forward(target_feature.float().cuda())
            # =========================generate transferable examples
            source_label_0 = source_label.cuda()
            feature_fooling_t = target_feature.float().cuda()
            feature_fooling_t.requires_grad = True
            feature_fooling_c = source_feature.float().cuda()
            feature_fooling_c.requires_grad = True
            feature_fooling_t_0 = feature_fooling_t.detach()
            
            for i in range(2):
                scores = discriminator.forward(feature_fooling_t)
                loss = BCELossForMultiClassification(torch.ones_like(scores) , 1 - scores) - 0.1 * torch.sum((feature_fooling_t - feature_fooling_t_0) * (feature_fooling_t - feature_fooling_t_0))
                loss.backward()
                g = feature_fooling_t.grad
                feature_fooling_t = feature_fooling_t + 0.001 * g 
                cls.zero_grad()
                discriminator.zero_grad()
                feature_fooling_t = feature_fooling_t.data.cpu().cuda()
                feature_fooling_t.requires_grad = True
                    
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
                _,_,_,scorec = cls.forward(feature_fooling_c)
                # _,_,scorec = cls.forward(feature_fooling_c)
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
            feature_source = source_feature.float().cuda()
            source_label = source_label.float().cuda()
            feature_target = target_feature.float().cuda()
            target_label = target_label.float().cuda()
            
            # _, _, predict_reg_source = cls.forward(feature_source)
            # _, _, predict_reg_target = cls.forward(feature_target)
            # _, _, predict_reg_fooling_t = cls.forward(feature_fooling_t)
            # _, _, predict_reg_fooling_c = cls.forward(feature_fooling_c)
            _, _, _, predict_reg_source = cls.forward(feature_source)
            _, _, _, predict_reg_target = cls.forward(feature_target)
            _, _, _, predict_reg_fooling_t = cls.forward(feature_fooling_t)
            _, _, _, predict_reg_fooling_c = cls.forward(feature_fooling_c)            

            domain_prob_source = discriminator.forward(feature_source)
            domain_prob_target = discriminator.forward(feature_target)
            domain_prob_fooling_t = discriminator.forward(feature_fooling_t)
            domain_prob_fooling_c = discriminator.forward(feature_fooling_c)
            
            dloss = BCELossForMultiClassification(torch.ones_like(domain_prob_source), domain_prob_source)
            dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_target), 1 - domain_prob_target)
            dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_fooling_c), domain_prob_fooling_c.detach())
            dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_fooling_t), 1 - domain_prob_fooling_t.detach())
            
            # ce = CrossEntropyLoss(source_label, predict_reg_source)
            # ce_extra_c = CrossEntropyLoss(source_label, predict_reg_fooling_c)
            ce = MSELoss(source_label, predict_reg_source)
            ce_extra_c = MSELoss(source_label, predict_reg_fooling_c)
            
            dis = torch.sum((predict_reg_fooling_t - predict_reg_target) * (predict_reg_fooling_t - predict_reg_target))
                

            with OptimizerManager([optimizer_cls , optimizer_discriminator]):
                loss = ce + ce_extra_c + dis + dloss
                loss.backward()

        k += 1
        # print the information
        print('epoch:{0}, domain loss:{1}, source loss:{2}, extra loss:{3}, dis loss:{4}'.format(k, dloss, ce, ce_extra_c, dis))
        # test
        source_r2, source_rmse, target_r2, target_rmse = test(test_source_dataset_name, test_source_label_name, test_target_dataset_name, test_target_label_name, mynet=[cls, discriminator, feature_extractor])
        if source_r2 > source_r2_max or source_rmse < source_rmse_min or target_r2 > target_r2_max or target_rmse < target_rmse_min:
            source_r2_max = source_r2
            source_rmse_min = source_rmse
            target_r2_max = target_r2
            target_rmse_min = target_rmse
            # save the model
            source_dataset_name = remove(source_dataset_name)
            target_dataset_name = remove(target_dataset_name)
            torch.save(cls, 'models/cls_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')
            torch.save(discriminator, 'models/discriminator_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')
            torch.save(feature_extractor, 'models/feature_extractor_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')
            continue
        else:
            break
        
    # save the model
    # source_dataset_name = remove(source_dataset_name)
    # target_dataset_name = remove(target_dataset_name)
    # torch.save(cls, 'models/cls_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')
    # torch.save(discriminator, 'models/discriminator_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')
    # torch.save(feature_extractor, 'models/feature_extractor_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')
    
    # print('test for {0} and {1} is finished'.format(test_source_dataset_name, test_target_dataset_name))
    # test(test_source_dataset_name, test_source_label_name, test_target_dataset_name, test_target_label_name)
    # print('train for {0} and {1} is finished'.format(source_dataset_name, target_dataset_name))

if __name__ == '__main__':
    train('train_X_kla240.mat', 'train_EQvec_kla240.mat', 'train_X_mu0.7.mat', 'train_EQvec_mu0.7.mat')