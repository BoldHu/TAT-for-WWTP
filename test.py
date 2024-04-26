import torch
from torch.utils.data import DataLoader
from torch.autograd.variable import *
import os
from collections import *
from utilities import *
from data import *
from networks import *
from data_loader import GetLoader
from torch.backends import cudnn
from reg_functions import reg_indicator
from remove_word import remove, change
import random

def test(source_feature, source_label, target_feature, target_label, mynet=None):
    
    # hyperparameters setting
    batch_size = 1344
    cuda = True
    cudnn.benchmark = True
    # set the random seed
    random.seed(42)
    torch.manual_seed(42)

    # =====================load data
    source_dataset_name = source_feature
    target_dataset_name = target_feature
    source_label_name = source_label
    target_label_name = target_label
    
    source_dataset = GetLoader(source_dataset_name, source_label_name)
    target_dataset = GetLoader(target_dataset_name, target_label_name)
    
    source_dataloader = DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    target_dataloader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    source_dataset_name = remove(source_dataset_name)
    target_dataset_name = remove(target_dataset_name)
    if mynet == None:
        # Load trained models
        cls = torch.load(os.path.join('models', 'cls_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')).cuda()
        discriminator = torch.load(os.path.join('models', 'discriminator_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')).cuda()
        feature_extractor = torch.load(os.path.join('models', 'feature_extractor_model_' + source_dataset_name + '_' + target_dataset_name + '.pt')).cuda()
    else:
        cls = mynet[0]
        discriminator = mynet[1]
        feature_extractor = mynet[2]

    # =====================test
    cls.eval()  # Set the classifier to evaluation mode
    discriminator.eval()  # Set the discriminator to evaluation mode
    
    source_mse = 0.0
    target_mse = 0.0
    n_samples = 0
    source_r2_list = []
    source_rmse_list = []
    target_r2_list = []
    target_rmse_list = []

    with torch.no_grad():  # No need to track gradients for testing
        for (source, target) in zip(source_dataloader, target_dataloader):
            source_feature, source_label = source
            target_feature, target_label = target
            
            # feature extract
            source_feature = feature_extractor.forward(source_feature.float().cuda())
            target_feature = feature_extractor.forward(target_feature.float().cuda())
            # Move data to GPU
            source_feature = source_feature.float().cuda()
            target_feature = target_feature.float().cuda()
            source_label = source_label.float().cuda()
            target_label = target_label.float().cuda()

            # Get predictions
            # _, _, predict_reg_source = cls(source_feature)
            # _, _, predict_reg_target = cls(target_feature)
            _, _, _, predict_reg_source = cls(source_feature)
            _, _, _, predict_reg_target = cls(target_feature)
            
            
            # Calculate MSE for source and target predictions
            source_mse += torch.sum((predict_reg_source - source_label) ** 2).item()
            target_mse += torch.sum((predict_reg_target - target_label) ** 2).item()
            n_samples += source_label.size(0) + target_label.size(0)
            # calculate the r2 and rmse for source and target data
            source_r2, source_rmse = reg_indicator(predict_reg_source, source_label)
            target_r2, target_rmse = reg_indicator(predict_reg_target, target_label)
            source_r2_list.append(source_r2.item())
            source_rmse_list.append(source_rmse.item())
            target_r2_list.append(target_r2.item())
            target_rmse_list.append(target_rmse.item())
    
    # Calculate the average r2 and rmse for source and target data
    source_r2 = sum(source_r2_list) / len(source_r2_list)
    source_rmse = sum(source_rmse_list) / len(source_rmse_list)
    target_r2 = sum(target_r2_list) / len(target_r2_list)
    target_rmse = sum(target_rmse_list) / len(target_rmse_list)
    # Print MSE for source and target
    print('MSE on the source data:{0}, MSE on the target data:{1}'.format(source_mse / n_samples, target_mse / n_samples))
    # print the r2 and rmse for source and target data
    print('R2 on the source data:{0:.4f}, RMSE on the source data:{1:.4f}, R2 on the target data:{2:.4f}, RMSE on the target data:{3:.4f}'.format(source_r2, source_rmse, target_r2, target_rmse))
    
    return source_r2, source_rmse, target_r2, target_rmse

if __name__ == '__main__':
    test('test_X_kla240.mat', 'test_EQvec_kla240.mat', 'test_X_mu0.7.mat', 'test_EQvec_mu0.7.mat')
