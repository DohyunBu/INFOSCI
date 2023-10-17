from torch import nn
from utils.utils import to_cuda
import torch

# intra_MMD - inter_MMD
class CDD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul,
                 num_classes, intra_only=False, **kwargs):

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes == 1)
        self.num_layers = num_layers

    # nums : class별 데이터 개수 [0번 class 데이터 개수, 1번 class 데이터 개수, ...]
    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            # dist_c : intra_data들의 상호간 거리 matrix
            dist_c = dist[start:end, start:end]
            # dist_list : [0번 class intra matrix, 1번 ...]
            dist_list += [dist_c]
        # dist_list : [0번 class data간의 거리 Matrix, 1번 class data간의 거리 Matrix, ...] 
        return dist_list


    def gamma_estimation(self, dist):

        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
                   2 * torch.sum(dist['st']) # Source + Target -> 서로 다른 두 데이터 거리 합

        # bs_S : source data의 class 개수
        bs_S = dist['ss'].size(0)  
        # bs_T : target data의 class 개수
        bs_T = dist['tt'].size(0)

        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T # N^2 - N : diagonal 값은 0 (같은 데이터 거리 = 0) batch_size
        gamma = dist_sum.item() / N # Source + Target -> E(서로 다른 두 데이터 거리)
        return gamma
    

    # nums_S, nums_T : class별 데이터 개수 리스트 형태 [0 class 데이터 개수, 1 class 데이터 개수, ...]
    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert (len(nums_S) == len(nums_T))
        num_classes = len(nums_S)

        patch = {}
        gammas = {}
        gammas['st'] = torch.zeros_like(dist['st'], requires_grad=False)
        gammas['ss'] = []
        gammas['tt'] = []
        for c in range(num_classes):
            gammas['ss'] += [(torch.zeros([num_classes], requires_grad=False))] # [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],...]
            gammas['tt'] += [(torch.zeros([num_classes], requires_grad=False))]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns] # patch['ss'] : ns class인 Source 데이터 사이의 거리

            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end
                target_end = target_start + nums_T[nt]
                patch['tt'] = dist['tt'][nt] # patch['tt'] : nt class인 Target 데이터 사이의 거리

                patch['st'] = dist['st'].narrow(0, source_start,
                                                nums_S[ns]).narrow(1, target_start, nums_T[nt]) # patch['st'] = Source ns class의 데이터와 Target nt class의 데이터 사이의 거리 

                gamma = self.gamma_estimation(patch) # Source ns class 데이터 + Target nt class 데이터 -> E(임의의 서로다른 두 데이터 간의 거리)

                # gammas['ss']T = gammas['tt'], gammas['ss']의 ns행 nt열 값 =  # Source ns class 데이터 + Target nt class 데이터 -> E(임의의 서로다른 두 데이터 간의 거리)
                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma
                # gammas['st'] = inter_MMD(s,t) : bs_S x bs_T matrix
                gammas['st'][source_start:source_end, \
                target_start:target_end] = gamma

        return gammas
    

    # input dist : MMD(domain의 class별 분포 ~ domain의 class별 분포 사이 거리)
    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        # base_gamma = gamma작음~gamma~gamma큼 : grid search
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        # gamma_grid : gamma_tensor
        gamma_list = [base_gamma * (kernel_mul ** i) for i in range(kernel_num)]
        gamma_tensor = (torch.stack(gamma_list, dim=0))

        # gamma_grid 중 너무 작은 애들은 eps으로 바꿔(너무 작은 gamma는 의미없어)
        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        # dist 차원 = gamma_tensor 차원
        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor # dist[i][j] / gammas[i][j]
        upper_mask = (dist > 1e5).type(torch.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        # dist[i][j] / gammas[i][j] -> upper보다 크면 10^5, lower보다 작으면 10^(-5), 중간에 있으면 dist[i][j] / gammas[i][j]
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        # kernel_val = rbf kernel 값
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    # 
    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = self.num_layers
        kernel_dist = None
        for i in range(num_layers):
            
            # MMD(domain의 class별 분포 ~ domain의 class별 분포 사이 거리)
            dist = dist_layers[i][key] if category is None else dist_layers[i][key][category]
            # gammas
            gamma = gamma_layers[i][key] if category is None else gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if kernel_dist is None:
                # kernel_dist = rbf kernel distance
                kernel_dist = self.compute_kernel_dist(dist, gamma, cur_kernel_num, cur_kernel_mul)
                continue
            
            # kernel_dist = rbf kernel distance가 layer 개수만큼 쌓임
            kernel_dist += self.compute_kernel_dist(dist, gamma, cur_kernel_num, cur_kernel_mul)

        return kernel_dist

    # nums_row = 1번 도메인 S/T [0번 클래스 데이터 개수, 1번 클래스 데이터 개수, ...], nums_col = 2번 도메인 S/T [0번 클래스 데이터 개수, 1번 클래스 데이터 개수, ...]
    def patch_mean(self, nums_row, nums_col, dist):
        assert (len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        mean_tensor = (torch.zeros([num_classes, num_classes]))
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                # val = E(1번 도메인의 row class 데이터와 2번 도메인의 col class 데이터 사이의 거리)
                val = torch.mean(dist.narrow(0, row_start,
                                             nums_row[row]).narrow(1, col_start, nums_col[col]))
                mean_tensor[row, col] = val
        return mean_tensor

    # domain1(A)와 domain2(B)에 속한 data 간의 거리 구하는 함수
    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand)) ** 2).sum(2)
        return dist


    def forward(self, source, target, nums_S, nums_T):
        assert (len(nums_S) == len(nums_T)), \
            "The number of classes for source (%d) and target (%d) should be the same." \
            % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        # compute the dist 
        dist_layers = []
        gamma_layers = []

        for i in range(self.num_layers):
            cur_source = source[i] # fs_i
            cur_target = target[i] # ft_i

            
            dist = {}
            # Source data 간의 거리
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            # Target data 간의 거리
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            # Source data ~ Target data 간의 거리
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)
            # dist['ss'][i] : Source i class 데이터 간의 거리 Matrix
            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            # dist['tt'][i] : Target i class 데이터 간의 거리 Matrix
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)
            
            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist : MMD!!!
        for i in range(self.num_layers):
            for c in range(num_classes):
                # i번째 dataset -> i번째 gammas -> c class에 대한 gamma값('ss' : Source c class data간 거리 평균, 'tt' : T..)
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)

        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.patch_mean(nums_S, nums_T, kernel_dist_st)

        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):
            kernel_dist_ss += [torch.mean(self.kernel_layer_aggregation(dist_layers,
                                                                        gamma_layers, 'ss', c).view(num_classes, -1),
                                          dim=1)]
            kernel_dist_tt += [torch.mean(self.kernel_layer_aggregation(dist_layers,
                                                                        gamma_layers, 'tt', c).view(num_classes, -1),
                                          dim=1)]

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes

        inter = None
        if not self.intra_only:
            inter_mask = ((torch.ones([num_classes, num_classes])
                                  - torch.eye(num_classes)).type(torch.BoolTensor))
            inter_mmds = torch.masked_select(mmds, inter_mask)
            inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))

        cdd = intra if inter is None else intra - inter
        return {'cdd': cdd, 'intra': intra, 'inter': inter}
