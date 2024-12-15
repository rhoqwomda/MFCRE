import torch
import torch.nn as nn
from torch.nn.functional import normalize




'''
Sample-Weighted Focal Contrastive (SWFC) Loss:
1. Divide training samples into positive and negative pairs to maximize 
inter-class distances while minimizing intra-class distances;
2. Assign more importance to hard-to-classify positive pairs;
3. Assign more importance to minority classes. 
'''
class SampleWeightedFocalContrastiveLoss(nn.Module):

    def __init__(self, temp_param, focus_param, sample_weight_param, dataset, class_counts, device):
        '''
        temp_param: control the strength of penalty on hard negative samples;
        focus_param: forces the model to concentrate on hard-to-classify samples;
        sample_weight_param: control the strength of penalty on minority classes;
        dataset: MELD or IEMOCAP.
        device: cpu or cuda. 
        '''
        super().__init__()
        
        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param
        self.dataset = dataset
        self.class_counts = class_counts
        self.device = device

        if self.dataset == 'MELD':
            self.num_classes = 7
        elif self.dataset == 'IEMOCAP':
            self.num_classes = 6
        else:
            raise ValueError('Please choose either MELD or IEMOCAP')
        
        self.class_weights = self.get_sample_weights()
    

    '''
    Use dot-product to measure the similarity between feature pairs.
    '''

    def dot_product_similarity(self, current_features, feature_sets):

        similarity = torch.sum(current_features * feature_sets, dim=-1)
        similarity_probs = torch.softmax(similarity / self.temp_param, dim=0)

        return similarity_probs

    '''
    Calculate the loss contributed from positive pairs.
    '''
    def positive_pairs_loss(self, similarity_probs):
        pos_pairs_loss = torch.mean(torch.log(similarity_probs) * ((1 - similarity_probs)**self.focus_param), dim = 0)

        return pos_pairs_loss


    '''
    Assign more importance to minority classes. 
    '''
    def get_sample_weights(self):
        total_counts = torch.sum(self.class_counts, dim = -1)
        # 计算所有类别样本的总数。
        class_weights = (total_counts / self.class_counts)**self.sample_weight_param
        # 根据类别样本的比例和权重参数（self.sample_weight_param）计算每个类别的权重。
        class_weights = normalize(class_weights, dim = -1, p = 1.0)
        # 对类别权重进行归一化，使其总和为1。

        return class_weights
        

    def forward(self, features, labels):
        self.num_samples = labels.shape[0]   # 确定批次中的样本数量。
        self.feature_dim = features.shape[-1]   # 确定特征向量的维数

        features = normalize(features, dim = -1)  # normalization helps smooth the learning process
        # 沿最后一个维度对特征向量进行归一化，以平滑学习过程，确保特征具有一致的尺度。

        batch_sample_weights = torch.FloatTensor([self.class_weights[label] for label in labels]).to(self.device)
        # 根据批次中每个样本的类别标签，为其创建样本权重的张量。
        # self.class_weights包含每个类别的预先计算的权重，并且它们由样本标签索引。
        # 将这些权重转换为张量并将它们移动到适当的设备（CPU 或 GPU）。

        total_loss = 0.0     # 将总损失初始化为零。
        for i in range(self.num_samples):     # 迭代批次中的每个样本以单独计算损失贡献。
            current_feature = features[i]
            current_label = labels[i]
            # 当前特征和标签
            feature_sets = torch.cat((features[:i], features[i + 1:]), dim = 0)
            # 创建一组不包括当前样本的特征。
            label_sets = torch.cat((labels[:i], labels[i + 1:]), dim = 0)
            # 创建一组不包括当前样本的标签。
            expand_current_features = current_feature.expand(self.num_samples - 1, self.feature_dim).to(self.device)
            similarity_probs = self.dot_product_similarity(expand_current_features, feature_sets)


            pos_similarity_probs = similarity_probs[label_sets == current_label]  # positive pairs with the same label
            if len(pos_similarity_probs) > 0:
                pos_pairs_loss = self.positive_pairs_loss(pos_similarity_probs)
                weighted_pos_pairs_loss = pos_pairs_loss * batch_sample_weights[i]
                total_loss += weighted_pos_pairs_loss
        
        loss = - total_loss / self.num_samples

        return loss