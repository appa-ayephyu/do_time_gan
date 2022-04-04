import torch.nn as nn
import torch
import numpy as np
from do_spatial_gan_pt.graph_gan.utils import BFS_trees, stable_softmax


class graph_gan_generator(nn.Module):
    def __init__(self, args, graph, n_node, node_emd_init):
        super(graph_gan_generator, self).__init__()
        self.args = args
        self.graph = graph

        self.BFS_trees = BFS_trees(
            self.root_nodes, self.graph, batch_num=self.args.cache_batch
        )

        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))

        self.node_embedding = None
        self.node_neighbor_embedding = None

        # parameters_list = self.embedding_matrix.para
        self.g_opt = torch.optim.Adam(
            [self.embedding_matrix, self.bias], lr=self.args.lr_gen
        )

    def all_score(self):
        return torch.matmul(
            self.embedding_matrix, torch.transpose(self.embedding_matrix, 0, 1)
        ).detach()

    def score(self, node_id, node_neighbor_id):
        """
        score: 一个n维向量，n为节点数。假设用向量g_v表示点v, 则g_v与各样本点v1的表示向量g_v1的内积组成的向量即为score
        """
        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        return (
            torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1)
            + self.bias[node_neighbor_id]
        )

    def loss(self, prob, reward):
        """
        Args:
            prob: D(Z)
            reward: 强化学习的奖励因子

        原始的生成器损失函数为 minimize mean(log(1-D(Z))), Z为负样本

        但是原始的损失函数无法提供足够梯度，导致生成器得不到训练

        作为替代，实际运行时使用的是 maximize mean(log(D(Z)))

        因此，对 -mean(log(D(Z))) 梯度下降即可
        """
        l2_loss = lambda x: torch.sum(x * x) / 2 * self.args.lambda_gen
        prob = torch.clamp(input=prob, min=1e-5, max=1)
        # 正则项
        regularization = l2_loss(self.node_embedding) + l2_loss(
            self.node_neighbor_embedding
        )
        _loss = -torch.mean(torch.log(prob) * reward) + regularization

        return _loss

    def preprocessing_it(self):
        # print()
        pass

    def train_it(self, data, discriminator):
        print()

        loss = self.forward(data, discriminator)
        self.g_opt.zero_grad()
        loss.backward()
        self.g_opt.step()

    def sample(self, root, tree, sample_num, discriminator=None):
        samples = []
        paths = []
        n = 0

        all_score = self.all_score()
        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = (
                    tree[current_node][1:] if is_root else tree[current_node]
                )
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = stable_softmax(relevance_probability)
                next_node = np.random.choice(
                    node_neighbor, size=1, p=relevance_probability
                )[
                    0
                ]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    def eval_it(self, data, discriminator):
        print()
        d_loss = self.forward(data, discriminator)
        return {"d_loss": d_loss.detach().cpu().numpy()}

    def forward(self, data, discriminator):
        root_nodes = data["root_nodes"]
        paths = []
        for i in range(root_nodes):
            isample, paths_from_i = self.sample(
                i, self.BFS_trees.get_tree(i), self.args.n_sample_gen
            )
            if paths_from_i is not None:
                paths.extend(paths_from_i)

        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])

        reward = discriminator.reward(node_1, node_2)

        # train_size = len(node_1)
        # start_list = list(range(0, train_size, self.args.batch_size_gen))
        # np.random.shuffle(start_list)

        score = self.score(
            node_id=np.array(node_1),
            node_neighbor_id=np.array(node_2),
        )
        prob = discriminator(score)
        loss = self.loss(prob=prob, reward=reward)

        return loss


class graph_gan_discriminator(nn.Module):
    def __init__(self, args, graph, n_node, node_emd_init):
        super(graph_gan_discriminator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.args = args
        self.graph = graph

        self.embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.neighboor_bias = None

        self.d_opt = torch.optim.Adam(
            [self.embedding_matrix, self.bias], lr=self.args.lr_dis
        )

    def preprocessing_it(self):
        # print()
        pass

    def train_it(self, data, generator=None):
        # print()
        d_loss = self.forward(data, generator)
        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()

    def eval_it(self, data, generator=None):
        # print()
        d_loss = self.forward(data, generator)
        return {"d_loss": d_loss.detach().cpu().numpy()}

    def forward(self, data, generator):
        # print()
        root_nodes = data["root_nodes"]
        center_nodes = []
        neighbor_nodes = []
        labels = []
        none_cnt = 0
        all_score = generator.all_score()
        for i in range(root_nodes):
            pos = self.graph[i]
            neg, _ = self.sample(i, self.BFS_trees.get_tree(i), len(pos), all_score)
            if neg is None:
                none_cnt += 1
            if len(pos) != 0 and neg is not None:
                # positive samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(pos)
                labels.extend([1] * len(pos))

                # negative samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(neg)
                labels.extend([0] * len(neg))

        center_nodes = data["center_nodes"]
        neighbor_nodes = data["neighbor_nodes"]
        labels = data["labels"]
        loss = self.loss(
            node_id=np.array(center_nodes),
            node_neighbor_id=np.array(neighbor_nodes),
            label=np.array(labels),
        )
        return loss

    def score(self, node_id, node_neighbor_id):
        """
        score: 一个n维向量，n为节点数。假设用向量d_v表示点v, 则d_v与各样本点v1的表示向量d_v1的内积组成的向量即为score
                表示向量的内积涵义为两节点相连的评分
        """
        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        self.neighboor_bias = self.bias[node_neighbor_id]
        # print(torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1).shape)
        # print(self.bias[node_neighbor_id].shape)
        return (
            torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1)
            + self.neighboor_bias
        )

    def loss(self, node_id, node_neighbor_id, label):
        """
        我们的目标是 maximize mean(log(D(x)) + log(1 - D(G(z))))

        因为BCEloss的结果为 -mean(log(D(x)) + log(1 - D(G(z))))

        所以直接对BCEloss的结果梯度下降即可
        """
        l2_loss = lambda x: torch.sum(x * x) / 2 * self.args.lambda_dis
        prob = torch.sigmoid(self.score(node_id, node_neighbor_id))
        criterion = nn.BCELoss()
        # 正则项
        regularization = (
            l2_loss(self.node_embedding)
            + l2_loss(self.node_neighbor_embedding)
            + l2_loss(self.neighboor_bias)
        )
        _loss = criterion(prob, torch.tensor(label).double()) + regularization
        return _loss

    def reward(self, node_id, node_neighbor_id):
        """
        强化学习，用于generator的训练
        """
        return torch.log(
            torch.tensor(1.0) + torch.exp(self.score(node_id, node_neighbor_id))
        ).detach()

    def sample(self, root, tree, sample_num, all_score):
        """sample nodes from BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            all_score
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = (
                    tree[current_node][1:] if is_root else tree[current_node]
                )
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                # if for_d:  # skip 1-hop nodes (positive samples)
                if node_neighbor == [root]:
                    # in current version, None is returned for simplicity
                    return None, None
                if root in node_neighbor:
                    node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = stable_softmax(relevance_probability)
                next_node = np.random.choice(
                    node_neighbor, size=1, p=relevance_probability
                )[
                    0
                ]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths
