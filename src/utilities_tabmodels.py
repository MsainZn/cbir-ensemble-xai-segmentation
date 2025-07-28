# Imports
import numpy as np
from scipy.spatial.distance import euclidean
import copy

# PyTorch Imports
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F

# Project Imports
from utilities_traintest import test_ndcg 



# Class: TabularMLP
class TabularMLP(nn.Module):

    # Method: __init__
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.bn2 = BatchNorm1d(2*hidden_dim) 
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim) 
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        return


    # Method: forward
    def forward(self, x):
        x = F.sigmoid(self.bn1(self.fc1(x)))
        x = F.sigmoid(self.bn2(self.fc2(x)))
        x = F.sigmoid(self.bn3(self.fc3(x)))
        x = F.sigmoid(self.fc4(x))
        
        return x
    

    # Method: get_transform
    def get_transform(self):
        def transform(tabular_vector):
            x = torch.tensor(tabular_vector, dtype=torch.float32).squeeze(0)
            return x
        return transform



# Dictionary: Models dictionary
MODELS_DICT = {
    "TabularMLP_773_200_20": TabularMLP(773, 200, 20),
    "TabularMLP_517_200_20": TabularMLP(517, 200, 20),
    "TabularMLP_293_100_20":TabularMLP(293, 100, 20),
    "TabularMLP_197_100_20":TabularMLP(197, 100, 20)
}



# Function: Weighted Euclidean
def weighted_euclidean_torch(v1, v2, W_mat, squared=True, abs_diff=True):
    """
    Calculate the Euclidean distance between two vectors using PyTorch.
    Args:
    v1, v2 (torch.Tensor): Input tensors.

    Returns:
    torch.Tensor: The Euclidean distance.
    """
    diff_vec  = torch.sub(v1, v2) # + 1e-15 
    diff_vecM = diff_vec.view(diff_vec.shape[0], 1)
    diff_vecT = diff_vec.view(1, diff_vec.shape[0])

    if abs_diff == True:
        diff_vecM = np.abs(diff_vecM)
        diff_vecT = np.abs(diff_vecT)

    if squared == True:
        W_mat = W_mat ** 2

    return torch.sqrt(torch.matmul(torch.matmul(diff_vecT, W_mat), diff_vecM))



# Function: Euclidean Model Manager
def eucl_model_manager_torch(v1, v2, W_mat, is_mat=False, dim=0):

    if is_mat == False and dim == 0:
        W = torch.diag(W_mat) + 1e-15
    elif dim != 0:
        W = torch.matmul(torch.transpose(W_mat, 0, 1), W_mat)
    else: 
        W = W_mat

    return weighted_euclidean_torch(v1, v2, W)



# Function: Euclidean Optimizer
def euclidean_optimizer(QNS_list_train, QNS_list_test, is_mat=False, dim=0, lr=0.0001, num_epochs=100, margin=0.00001):

    qns = copy.deepcopy(QNS_list_train)
    num_features = len(qns[0].query_vector)
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()
    
    # Configuring Tthe weights based on the matrix format
    if is_mat==True:
        # weights = torch.eye(num_features, dtype=torch.float64) + 1e-15
        weights = torch.rand(num_features, num_features) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)
    else:    
        # weights = torch.ones(num_features, dtype=torch.float64) + 1e-15
        weights = torch.rand(num_features, dtype=torch.float64) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)
    
    if dim != 0:
        weights = torch.rand(dim, num_features) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)

    # print("Initial weights:\n", weights)
    optimizer = torch.optim.Adam([weights], lr=lr)

    final_ordering = []
    for epoch in range(0, num_epochs):
        optimizer.zero_grad()
        success_count = 0
        total_count = 0
        loss = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        for q_element in qns:
            qn_dist = []
            for i in range(q_element.neighbor_count):
                for j in range(i + 1, q_element.neighbor_count):
                    dist_i = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, is_mat, dim)
                    dist_j = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, is_mat, dim)
                    
                    cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                    # print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {q_element.score_vector[i].item():02.0f}, Score-j: {q_element.score_vector[j].item():02.0f} Update-Loss: {cond.item()}')
                    if cond:
                        loss = loss + torch.abs(dist_i - dist_j) + 1.00 / (torch.sum(torch.abs(weights)) + margin)
                        # loss = loss + F.relu(margin + (dist_i - dist_j)) / (torch.sum(weights) + margin)
                    else:
                        success_count = success_count + 1
                    total_count = total_count + 1
                qn_dist.append(dist_i.item())
            final_ordering.append(qn_dist)
            
        acc_train = success_count/total_count
        ndcg_train = 100 * np.mean(test_ndcg(final_ordering))
        final_ordering.clear()

        copied_tensor = weights.detach().clone().numpy()
        acc_test, ndcg_test = euclidean_evaluate(QNS_list_test, copied_tensor, is_mat=is_mat, dim=dim)
        
        # print(f'Epoch {epoch} loss is: {loss.item():.10f} Train-Acc: {acc_train:.6} Train-nDCG: {ndcg_train:.6} Test-Acc: {acc_test:.6} Test-nDCG: {ndcg_test:.6}')

        loss.backward()
        optimizer.step()

    # print('Summary:')
    copied_tensor = weights.detach().clone().numpy()
    acc_base_train, ndcg_base_train = euclidean_base(QNS_list_train)
    acc_train, ndcg_train = euclidean_evaluate(QNS_list_train, copied_tensor, is_mat=is_mat, dim=dim)
    # print(f'Trainset Raw-Euclidian Acc: {acc_base_train:.6} and nDCG: {ndcg_base_train:.6} | Model-Acc: {acc_train:.6} Model-nDCG: {ndcg_train:.6}!')
    # print(f'Trainset Rate of Improvement: Acc: {100*(acc_train-acc_base_train)/acc_base_train:.6}% and nDCG: {100*(ndcg_train-ndcg_base_train)/ndcg_base_train:.6}%')
    
    acc_base_test, ndcg_base_test = euclidean_base(QNS_list_test)
    acc_test, ndcg_test = euclidean_evaluate(QNS_list_test, copied_tensor, is_mat=is_mat, dim=dim)
    # print(f'Testset Raw-Euclidian Acc: {acc_base_test:.6} and nDCG: {ndcg_base_test:.6} | Model-Acc: {acc_test:.6} Model-nDCG: {ndcg_test:.6}!')
    # print(f'Testset Rate of Improvement: Acc: {100*(acc_test-acc_base_test)/acc_base_test:.6}% and nDCG: {100*(ndcg_test-ndcg_base_test)/ndcg_base_test:.6}%')
    
    return weights.detach().numpy()



# Function: Euclidean Evaluate (not adjusted, deprecated)
def _euclidean_evaluate(QNS_list, iweights, is_mat, dim=0):

    qns = copy.deepcopy(QNS_list)

    # Preparing Testset into right format
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()
    
    # Preparing the weights into right format
    weights = torch.from_numpy(iweights)
    weights = weights.type(torch.float64)

    # Accuracy Evaluation
    final_ordering = []
    success_count = 0
    total_count = 0

    # Evaluation Loop
    for q_element in qns:
        qn_dist = []
        for i in range(q_element.neighbor_count):
            dist_i = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, is_mat, dim)
            for j in range(i + 1, q_element.neighbor_count):
                dist_j = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, is_mat, dim)
                cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                if cond == False:
                    success_count = success_count + 1   
                total_count = total_count + 1
            qn_dist.append(dist_i.item())
        final_ordering.append(qn_dist)
    
    acc = success_count/total_count
    ndcg = 100 * np.mean(test_ndcg(final_ordering))
    
    return acc, ndcg



# Function: Euclidean Evaluate (adjusted, current version)
def euclidean_evaluate(QNS_list, iweights, is_mat, dim=0):

    qns = copy.deepcopy(QNS_list)

    # Preparing test set into right format
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()

    # Preparing the weights into right format
    weights = torch.from_numpy(iweights)
    weights = weights.type(torch.float64)

    # Accuracy Evaluation
    final_ordering = []
    rev_ordering = []
    success_count = 0
    total_count = 0

    # Evaluation Loop
    for q_element in qns:
        qn_dist = []
        rs_dist = []
        sz = len(q_element.neighbor_vectors)
        count = 0
        for i in range(q_element.neighbor_count):
            dist_i = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, is_mat, dim)
            for j in range(i + 1, q_element.neighbor_count):
                dist_j = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, is_mat, dim)
                cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                if cond == False:
                    success_count = success_count + 1   
                total_count = total_count + 1

            qn_dist.append(dist_i.item())
            rs_dist.append(sz - count)
            count += 1
        final_ordering.append(qn_dist)
        rev_ordering.append(rs_dist)

    acc = success_count/total_count
    ndcg = 100 * np.mean((test_ndcg(final_ordering) - test_ndcg(rev_ordering))/(1 - test_ndcg(rev_ordering)))

    return acc, ndcg



# Function: Euclidean Base
def euclidean_base(qns):

    # Accuracy Evaluation
    final_ordering = []
    success_count = 0
    total_count = 0

    # Evaluation Loop
    for q_element in qns:
        qn_dist = []
        for i in range(q_element.neighbor_count):
            dist_i = euclidean(q_element.query_vector, q_element.neighbor_vectors[i])
            for j in range(i + 1, q_element.neighbor_count):
                dist_j = euclidean(q_element.query_vector, q_element.neighbor_vectors[j])
                cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                if cond == False:
                    success_count = success_count + 1   
                total_count = total_count + 1
            qn_dist.append(dist_i)
        final_ordering.append(qn_dist)
    
    acc = success_count/total_count
    ndcg = 100 * np.mean(test_ndcg(final_ordering))
    
    return acc, ndcg



# Function: Collaborative Filtering Tabular Normalize
def collaborative_tabular_normalize(qns_list, min_max_values=None):

    if min_max_values is not None:
        vec_len = len(min_max_values)
    else:
        vec_len = len(qns_list[0].query_vector)
        min_max_values = []

    all_elements = [[] for _ in range(vec_len)]

    # Collecting all elements for each position from both query and neighbor vectors
    for qns in qns_list:
        for i in range(vec_len):
            all_elements[i].append(qns.query_vector[i])
            for neighbor_vector in qns.neighbor_vectors:
                all_elements[i].append(neighbor_vector[i])
    
    # If min_max_values is provided, use it for normalization
    if min_max_values:
        for i in range(vec_len):
            min_val, max_val = min_max_values[i]
            all_elements[i] = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in all_elements[i]]
    
    else:
        # Normalizing each position across all instances and storing min-max values
        for i in range(vec_len):
            min_val = np.min(all_elements[i])
            max_val = np.max(all_elements[i])
            all_elements[i] = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in all_elements[i]]
            min_max_values.append((min_val, max_val))


    # Updating the vectors in QNS_structure instances
    for qns in qns_list:
        for i in range(vec_len):
            qns.query_vector[i] = all_elements[i].pop(0)
            for neighbor_vector in qns.neighbor_vectors:
                neighbor_vector[i] = all_elements[i].pop(0)

    return min_max_values