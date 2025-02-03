import torch
from torch.utils.data import DataLoader
import numpy as np
from impl import train_client
from meta_define.attack import Strategic
import util.easy_fed as ef

class DefaultStrategic(Strategic):

    def examine(self, context: dict) -> list:
        gradients = context['gradients']
        n = len(gradients)
        return [0 for _ in range(n)]

    def aggregate(self, context: dict):
        return context['updates']


class Krum(Strategic):

    def __init__(self):
        super().__init__()
        self.scores = None
        self.f = 3
        self.n = 10
        self.k = self.n - self.f - 1

    def get_score(self, g, gradients):
        s = []
        for grad in gradients:
            w = ef.l2(g, grad)
            s.append(w)
        list.sort(s)
        return sum(s[0:self.n - self.f])

    def update_grad_score(self, gradients):
        scores = []
        for i in range(len(gradients)):
            score = self.get_score(gradients[i], gradients)
            scores.append([score, i])
        scores.sort(key=lambda x: x[0])
        self.scores = scores

    def examine(self, context: dict) -> list:
        gradients = context['gradients']
        result = [1 for _ in range(len(gradients))]
        for i in range(self.k):
            result[self.scores[i][1]] = 0
        return result

    def aggregate(self, context: dict):
        updates = context['updates']
        n = len(updates)
        self.update_grad_score(updates)
        result = []
        for i in range(self.k):
            result.append(ef.mul_constant(updates[self.scores[i][1]], n / self.k))
        return result


class FLtrust(Strategic):
    def __init__(self):
        super().__init__()
        self.scale = 1
        self.update = None
        self.root_data, self.model_state, self.optim, self.lr, self.loss_fun, self.model_type, self.device \
            = None, None, None, None, None, None, None
        self.epoch = 1

    def assign(self, context: dict):
        train_dataset, _ = context["dataset"]
        data_size = 300
        self.root_data = DataLoader(dataset=torch.utils.data.Subset(train_dataset, range(0, data_size)),
                                    batch_size=min(32, data_size),
                                    shuffle=True)
        self.model_type = context["model"]
        self.optim, self.lr, self.loss_fun, self.device = \
            (context["optim"], context["lr"], context["loss_func"], context["device"])

    def train(self):
        if (self.root_data and self.model_state and self.optim and self.lr and self.loss_fun and self.model_type and
                self.device is not None):
            self.update = ef.copy(self.model_state)
            for i in range(self.epoch):
                update, sum_loss, true_predict, size = train_client(self.root_data, self.model_type, self.optim,
                                                                    self.lr,
                                                                    self.loss_fun, self.model_state, self.device)
                self.model_state = ef.sub(self.model_state, update)
            self.update = ef.mul_constant(ef.sub(self.update, self.model_state), self.scale)
            self.model_state = None

    def examine(self, context: dict) -> list:
        self.train()
        gradients = context['gradients']
        result = []
        for g in gradients:
            w, g_l2, go_l2 = ef.cos(g, self.update)
            if w <= 0:
                result.append(1)
            else:
                result.append(0)
        return result

    def aggregate(self, context: dict):
        model_state = context['global_model'].state_dict()
        self.model_state = ef.copy(model_state)
        self.train()
        updates = context['updates']
        result = []
        n = len(updates)
        sum_weight = 0
        for up in updates:
            w, g_l2, go_l2 = ef.cos(up, self.update)
            w = max(w, 0)
            sum_weight += w
            result.append(ef.mul_constant(up, w * go_l2 / g_l2))
        for i in range(len(result)):
            result[i] = ef.mul_constant(result[i], n)
        return result


# Dnc 算法实现
class DivideAndConquer(Strategic):
    def __init__(self, iters=1, subsample_size=1000, c=0.2):
        super().__init__()
        self.name = "DnC"
        self.iters = iters
        self.subsample_size = subsample_size
        self.c = c

    def _preprocess_gradients(self, gradients):
        # print(gradients)
        # for grad in gradients:
        #  print(grad["weight"].shape)
        grads = torch.stack([
            torch.cat([param.flatten() for param in grad.values()])
            for grad in gradients
        ], dim=0)
        grads[torch.isnan(grads)] = 0  # Replace NaN values with 0
        return grads

    def _subsample_and_compute_scores(self, grads, num_users):
        num_params = grads.shape[1]
        self.subsample_size = min(self.subsample_size, num_params)
        subsampled_indices = np.random.choice(range(num_params), self.subsample_size, replace=False)
        grads_subsampled = grads[:, subsampled_indices]

        mu = torch.mean(grads_subsampled, dim=0)
        grads_centered = grads_subsampled - mu

        _, _, V = torch.svd(grads_centered)
        v = V[:, 0]
        # Compute outlier scores
        outlier_scores = torch.sum((grads_subsampled - mu) * v, dim=1) ** 2
        return outlier_scores

    def aggregate(self, context: dict):
        gradients = context["updates"]
        # len(gradients) 有 客户端数量的字典  key value 形式
        # key
        num_users = len(gradients)
        num_byzantine = int(self.c * num_users)
        good_indices = set(range(num_users))
        grads = self._preprocess_gradients(gradients)

        for _ in range(self.iters):
            outlier_scores = self._subsample_and_compute_scores(grads, num_users)
            num_to_keep = num_users - num_byzantine
            top_indices = torch.argsort(outlier_scores)[:num_to_keep].cpu().numpy()
            good_indices = good_indices.intersection(set(top_indices))

        final_indices = list(good_indices)
        # 过滤之后的梯度 组成字典
        filtered_gradients = [gradients[i] for i in final_indices]
        # 返回的是 一个list  里面包含挑选下来剩余的客户端
        return filtered_gradients

    def examine(self, context: dict) -> list:
        gradients = context["gradients"]
        num_users = len(gradients)
        num_byzantine = int(self.c * num_users)
        grads = self._preprocess_gradients(gradients)
        outlier_scores = self._subsample_and_compute_scores(grads, num_users)
        num_to_flag = num_byzantine
        flagged_indices = torch.argsort(outlier_scores, descending=True)[:num_to_flag].cpu().numpy()
        flags = [0] * num_users
        for idx in flagged_indices:
            flags[idx] = 1
        print("flags", flags)
        return flags


class LayerMatrixDnc(Strategic):
    def __init__(self, iters = 1, c = 0.2, flags = None):
        super().__init__()
        self.flags = flags
        self.name = "layer_matrix_dnc"
        self.iters = iters
        self.c = c

    def matrix_dnc_aggregation(self, client_matrices, num_clients, c, n_iters):
        """
        矩阵形式的 Divide-and-Conquer 聚合算法（分层聚合版本）。
        Args:
            client_matrices: List[torch.Tensor]，每个客户端上传的矩阵 (m x n)。
            num_clients: 客户端数量。
            c: 恶意客户端比例。
            n_iters: 迭代次数。
        Returns:
            good_indices: 保留的正常客户端索引。
        """
        good_clients = set(range(num_clients))

        for _ in range(n_iters):
            # 计算所有客户端矩阵的均值矩阵
            mean_matrix = sum(client_matrices) / len(client_matrices)
            # 计算中心化矩阵
            centered_matrices = [matrix - mean_matrix for matrix in client_matrices]
            # 叠加中心化矩阵  这里采用叠加的操作
            stacked_matrix = sum(centered_matrices)
            # 计算叠加矩阵的 SVD
            #print(stacked_matrix.shape)
            U, S, Vh = torch.linalg.svd(stacked_matrix, full_matrices=False)
            top_singular_vector = Vh[0, :]  # 提取第一右奇异向量

            # 计算每个矩阵的异常分数
            outlier_scores = [
                torch.norm(torch.mm(centered_matrix, top_singular_vector.unsqueeze(1))) ** 2
                for centered_matrix in centered_matrices
            ]
            # 转换为张量
            outlier_scores = torch.tensor(outlier_scores)
            # 根据异常分数筛选正常客户端
            num_to_keep = int((1 - c) * num_clients)
            # print(num_to_keep)
            top_indices = np.argsort(outlier_scores.cpu().numpy())[:num_to_keep]
            good_clients = good_clients.intersection(set(top_indices))

        return good_clients

    def vector_dnc_aggregation(self, client_vectors, num_clients, c, n_iters):
        good_clients = set(range(num_clients))  # 初始化良性客户端的集合

        for _ in range(n_iters):
            # 1. 拼接所有客户端的向量为一个矩阵，每个客户端的向量作为矩阵的一行
            stacked_matrix = torch.stack(client_vectors)

            # 2. 计算均值向量并中心化每个客户端的向量
            mean_vector = torch.mean(stacked_matrix, dim=0)
            centered_vectors = [vector - mean_vector for vector in client_vectors]

            # 3. 叠加中心化向量
            stacked_centered_vectors = torch.stack(centered_vectors)

            # 4. 计算叠加矩阵的 SVD
            _, _, Vh = torch.linalg.svd(stacked_centered_vectors, full_matrices=False)
            top_singular_vector = Vh[0, :]  # 提取第一右奇异向量

            # 5. 计算每个向量的异常分数
            # 使用矩阵乘法计算每个向量与 top_singular_vector 的投影，获取异常分数
            outlier_scores = torch.sum((stacked_centered_vectors @ top_singular_vector.unsqueeze(1)) ** 2, dim=1)

            # 6. 根据异常分数筛选正常客户端
            num_to_keep = int((1 - c) * num_clients)
            #print(num_to_keep)  # 输出需要保留的客户端数量

            # 获取异常分数最小的前 num_to_keep 个客户端
            top_indices = np.argsort(outlier_scores.cpu().numpy())[:num_to_keep]
            good_clients = good_clients.intersection(set(top_indices))

        return good_clients

    def aggregate(self, context: dict):
        gradients = context["updates"]
        num_users = len(gradients)
        # 对每个客户端的参数进行 reshape
        weights = []
        for client in gradients:
            weights.append(ef.model_select(client, "weight"))
        # 获取层数
        num_layers = len(weights[0])
        print("num_layers", num_layers)
        weights = ef.reshape_weights(weights)
        # 初始化标记数组
        self.flags = [0] * num_users
        client_layer_count = [0] * num_users  # 用于统计每个客户端被标记为异常的层数
        # 分层处理
        for layer_idx in weights[0].keys():  # 遍历每一层
            client_matrices = [client[layer_idx] for client in weights]  # 提取每个客户端的层矩阵
            # 获取良性客户端索引
            if client_matrices[0].dim() == 1:  # 1D vector (e.g., a weight vector)
                # Perform DNC aggregation for vectors
                good_indices = self.vector_dnc_aggregation(client_matrices, num_users, self.c, 10)
            else:  # Multidimensional matrix (e.g., weight matrices for layers)
                # Perform DNC aggregation for matrices
                good_indices = self.matrix_dnc_aggregation(client_matrices, num_users, self.c, self.iters)
            #good_indices = self.matrix_dnc_aggregation(client_matrices, num_users, self.c, self.iters)
            # 将异常客户端的该层参数置为 0
            #print("layer_idx", layer_idx)
            #print("good_indices", good_indices)
            # for i in range(num_users):
            #     if i not in good_indices:
            #         client_layer_count[i] += 1
            for i in range(num_users):
                if i not in good_indices:
                    gradients[i][layer_idx] = torch.zeros_like(gradients[i][layer_idx])
                    client_layer_count[i] += 1
        for i in range(num_users):
            if client_layer_count[i] > num_layers // 2:
                self.flags[i] = 1

        good_gradients = [gradients[i] for i in range(num_users) if self.flags[i] == 0]
        return good_gradients


    def examine(self, context: dict) -> list:
        # gradients = context["gradients"]
        # num_users = len(gradients)
        #
        # # print(num_layers)
        # # 对每个客户端的参数进行 reshape
        # weights = []
        # for client in gradients:
        #     weights.append(ef.model_select(client, "weight"))
        # # 获取层数
        # num_layers = len(weights[0])
        # weights = ef.reshape_weights(weights)
        #
        # # 初始化标记数组
        # flags = [0] * num_users
        # client_layer_count = [0] * num_users  # 用于统计每个客户端被标记为异常的层数
        #
        # # 分层处理
        # for layer_idx in weights[0].keys():  # 遍历每一层
        #     client_matrices = [client[layer_idx] for client in weights]  # 提取每个客户端的层矩阵
        #     # 获取良性客户端索引
        #     good_indices = self.matrix_dnc_aggregation(client_matrices, num_users, self.c, self.iters)
        #     # 标记异常客户端层数
        #     for i in range(num_users):
        #         if i not in good_indices:
        #             client_layer_count[i] += 1
        # # 如果客户端被标记为异常的层数超过一半，则标记为有毒
        # for i in range(num_users):
        #     if client_layer_count[i] > num_layers // 2:
        #         flags[i] = 1
        print("flags",self.flags)
        return self.flags


class Gpf(Strategic):
    def __init__(self, iters=2, subsample_size=500, c=0.3,k = 5):
        super().__init__()
        self.name = "Gpf"
        self.iters = iters
        self.subsample_size = subsample_size
        self.c = c
        self.k = k
    @staticmethod
    def random_projection(self, grads, target_dim):
        num_params = grads.size(1)
        if num_params <= target_dim:
            return grads

        # 生成随机正交矩阵
        random_matrix = torch.randn(num_params, target_dim, device=grads.device)
        q, _ = torch.linalg.qr(random_matrix)  # 正交化

        # 投影到低维空间
        return torch.matmul(grads, q)

    @staticmethod
    def compute_outlier_scores(self, grads):

        # 计算所有客户端在这些维度上的梯度均值
        mu = torch.mean(grads, dim=0)

        # 计算中心化梯度
        grads_centered = grads - mu

        # 通过 SVD 分解计算中心化梯度的主方向（最右奇异向量）
        _, _, V = torch.svd(grads_centered)
        v = V[:, 0]

        # 计算每个客户端的异常分数（在主方向上的投影平方和）
        outlier_scores = torch.sum((grads_centered @ v) ** 2, dim=1)

        return outlier_scores

    def aggregate(self, context: dict):
        # 从 context 中获取所有客户端上传的梯度
        gradients = context["updates"]
        num_users = len(gradients)
        num_byzantine = int(self.c * num_users)  # 估计恶意客户端数量
        good_indices = set(range(num_users))  # 初始化良性客户端索引集合

        # 将所有客户端的梯度扁平化为一个大矩阵
        grads = torch.stack([
            torch.cat([param.flatten() for param in grad.values()])
            for grad in gradients
        ], dim=0)
        num_params = grads.shape[1]  # 每个客户端展平后的梯度长度
        grads[torch.isnan(grads)] = 0  # 处理 NaN 值

        # 确保 subsample_size 不超过梯度总维度数
        target_dim = min(self.subsample_size, num_params)

        for _ in range(self.iters):
            # 使用正交矩阵降维
            grads_subsampled = self.random_projection(grads, target_dim)

            # 计算每个客户端的异常分数
            outlier_scores = self.compute_outlier_scores(grads_subsampled)

            # 选取异常分数最低的 n - c * m 个客户端
            num_to_keep = num_users - num_byzantine
            top_indices = torch.argsort(outlier_scores)[:num_to_keep].cpu().numpy()
            good_indices = good_indices.intersection(set(top_indices))

        # 过滤后的客户端梯度集合
        final_indices = list(good_indices)
        filtered_gradients = [gradients[i] for i in final_indices]

        return filtered_gradients

    def examine(self, context: dict) -> list:
        # 从 context 中获取所有客户端上传的梯度
        gradients = context["gradients"]
        num_users = len(gradients)
        num_byzantine = int(self.c * num_users)  # 估计恶意客户端数量

        # 将所有客户端的梯度扁平化为一个大矩阵
        grads = torch.stack([
            torch.cat([param.flatten() for param in grad.values()])
            for grad in gradients
        ], dim=0)
        num_params = grads.shape[1]  # 每个客户端展平后的梯度长度
        grads[torch.isnan(grads)] = 0  # 处理 NaN 值

        # 确保 subsample_size 不超过梯度总维度数
        target_dim = min(self.subsample_size, num_params)

        # 使用正交矩阵降维
        grads_subsampled = self.random_projection(grads, target_dim)

        # 计算每个客户端的异常分数
        outlier_scores = self.compute_outlier_scores(grads_subsampled)

        # 找到异常分数最高的 c * m 个客户端，并标记为有毒
        num_to_flag = num_byzantine
        flagged_indices = torch.argsort(outlier_scores, descending=True)[:num_to_flag].cpu().numpy()

        # 初始化标记数组，默认全为 0（无毒）
        flags = [0] * num_users
        for idx in flagged_indices:
            flags[idx] = 1

        return flags

# 基于马氏距离的聚合方法。
class SimpleMahalanobis(Strategic):
    def __init__(self, c=0.3, epsilon=1e-6, reduced_dim=500):
        super().__init__()
        self.name = "Mahalanobis"
        self.c = c  # 过滤比例
        self.epsilon = epsilon  # 协方差矩阵正则化参数
        self.reduced_dim = reduced_dim  # 降维后的目标维度

    def random_projection(self, gradients):

        num_params = gradients.size(1)
        if num_params <= self.reduced_dim:
            return gradients  # 如果参数维度已经小于目标维度，则无需降维

        # 生成随机正交矩阵
        random_matrix = torch.randn(num_params, self.reduced_dim, device=gradients.device)
        q, _ = torch.linalg.qr(random_matrix)  # 正交化

        # 投影到低维空间
        return torch.matmul(gradients, q)

    def compute_mahalanobis_distance(self, gradients):
        # 使用随机正交矩阵降维
        gradients = self.random_projection(gradients)

        # 计算均值向量和中心化后的梯度矩阵
        mu = torch.mean(gradients, dim=0)
        centered_grads = gradients - mu

        # 计算协方差矩阵及其逆
        cov_matrix = torch.mm(centered_grads.T, centered_grads) / (gradients.size(0) - 1)
        cov_matrix += self.epsilon * torch.eye(cov_matrix.size(0), device=gradients.device)
        cov_matrix_inv = torch.linalg.inv(cov_matrix)

        # 计算马氏距离
        mahalanobis_distances = torch.sqrt(
            torch.sum((centered_grads @ cov_matrix_inv) * centered_grads, dim=1)
        )

        return mahalanobis_distances

    def filter_gradients(self, gradients):
        distances = self.compute_mahalanobis_distance(gradients)
        threshold = torch.quantile(distances, 1 - self.c)  # 动态阈值
        good_indices = torch.where(distances <= threshold)[0].tolist()
        print(good_indices)
        filtered_gradients = gradients[good_indices]
        return filtered_gradients, good_indices

    def aggregate(self, context: dict) -> list:
        # 从 context 中获取所有客户端上传的梯度
        gradients = context["updates"]
        grads = self.flatten_gradients(gradients)

        # 过滤异常梯度
        filtered_gradients, good_indices = self.filter_gradients(grads)

        # 返回过滤后的客户端梯度
        return [gradients[i] for i in good_indices]

    def examine(self, context: dict) -> list:
        # 从 context 中获取所有客户端上传的梯度
        gradients = context["gradients"]
        grads = self.flatten_gradients(gradients)

        # 计算每个客户端的马氏距离
        distances = self.compute_mahalanobis_distance(grads)
        threshold = torch.quantile(distances, 1 - self.c)  # 动态阈值

        # 判断每个客户端是否异常
        return [1 if dist > threshold else 0 for dist in distances]

    @staticmethod
    def flatten_gradients(gradients):

        return torch.stack([
            torch.cat([param.flatten() for param in grad.values()])
            for grad in gradients
        ], dim=0)