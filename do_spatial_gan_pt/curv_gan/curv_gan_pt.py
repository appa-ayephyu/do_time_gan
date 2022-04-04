import torch
import torch.nn as nn
from do_spatial_gan_pt.curv_gan.models import Generator, Discriminator
import torch.nn.functional as F
import numpy as np
from do_spatial_gan_pt.curv_gan.utils import random_walk
import random


class curv_gan_discriminator(nn.Module):
    def __init__(self, args, graph):
        super(curv_gan_discriminator, self).__init__()
        self.args = args
        self.graph = graph
        self.discriminator = Discriminator(args)
        self.optimizer_d = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=args.lr_d,
            weight_decay=args.weight_decay_d,
        )
        self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optimizer_d,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma),
        )

    def preprocessing_it(self):
        pass

    def train_it(self, data, generator):
        print()
        g_loss = self.forward(data, generator)
        self.optimizer_d.zero_grad()
        g_loss.backward()
        self.optimizer_d.step()
        self.lr_scheduler_d.step()

    def eval_it(self, data, generator):
        g_loss = self.forward(data, generator)
        return {"g_loss": g_loss}

    def forward(self, data, generator):
        node_list = data["nodelist"]

        pos_node_ids = []
        pos_node_neighbor_ids = []
        neg_node_neighbor_ids = []
        for node_id in node_list:
            for k in range(self.config.walk_num):
                walk = random_walk(
                    node_id, graph=self.graph, walk_len=self.args.walk_len
                )
                for t in walk:
                    pos_node_ids.append(node_id)
                    pos_node_neighbor_ids.append(t)
                    neg = random.choice(self.nodelist)
                    neg_node_neighbor_ids.append(neg)

        # generate fake node()
        node_fake_embedding = generator(pos_node_ids).detach()

        pos_loss = F.binary_cross_entropy(
            self.discriminator(pos_node_ids, pos_node_neighbor_ids),
            torch.ones(len(pos_node_ids)).to(self.config.device),
        )
        neg_loss = F.binary_cross_entropy(
            self.discriminator(pos_node_ids, neg_node_neighbor_ids),
            torch.zeros(len(pos_node_ids)).to(self.config.device),
        )
        fake_loss = F.binary_cross_entropy(
            self.discriminator(pos_node_ids, node_fake_embedding),
            torch.zeros(len(pos_node_ids)).to(self.config.device),
        )
        g_loss = pos_loss + neg_loss + fake_loss
        return g_loss


class curv_gan_generator(nn.Module):
    def __init__(self, args, graph):
        super(curv_gan_generator, self).__init__()
        self.args = args
        self.graph = graph
        self.generator = Generator(args)
        # Optimizers
        if args.lr_reduce_freq is None:
            args.lr_reduce_freq = args.max_epochs
        self.optimizer_g = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=args.lr_g,
            weight_decay=args.weight_decay_g,
        )
        self.lr_scheduler_g = torch.optim.lr_scheduler.StepLR(
            self.optimizer_g,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma),
        )

    def preprocessing_it(self):
        pass

    def train_it(self, data, discriminator):
        g_loss = self.forward(data, discriminator)
        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()
        self.lr_scheduler_g.step()

    def eval_it(self, data, discriminator):
        g_loss = self.forward(data, discriminator)

        return {"g_loss": g_loss.detach().cpu().numpy()}

    def forward(self, data, discriminator):
        node_list = data["nodelist"]

        pos_node_ids = []
        pos_node_neighbor_ids = []
        for node_id in node_list:
            for k in range(self.args.walk_num):
                walk = random_walk(
                    node_id, graph=self.graph, walk_len=self.args.walk_len
                )
                for t in walk:
                    pos_node_ids.append(node_id)
                    pos_node_neighbor_ids.append(t)

        node_fake_embedding = self.generator(pos_node_ids)
        pos_node_embedding = self.generator.embedding.index_select(
            0, torch.tensor(pos_node_ids).to(self.args.device)
        )
        neighbor_fake_embedding = self.generator(pos_node_neighbor_ids)

        # 计算distortion
        if self.args.lmda > 0:
            dt_loss = self.distortion(
                pos_node_ids,
                pos_node_neighbor_ids,
                neighbor_fake_embedding,
                discriminator=discriminator,
            )
        else:
            dt_loss = torch.tensor(0.0)
        fake_loss = F.binary_cross_entropy(
            discriminator.discriminator(pos_node_embedding, node_fake_embedding),
            torch.ones(len(pos_node_ids)).to(self.args.device) * self.args.label_smooth,
        )

        d_loss = fake_loss + self.args.lmda * dt_loss
        return d_loss

    def distortion(self, u, pos_u, pos_fake_emb, discriminator):
        """
        Return L2 norm between true distortion and fake distortion.
        """
        # g_emb = self.generator.embedding
        d_emb = discriminator.discriminator.embedding
        nodes_list = np.array([u, pos_u]).T
        distortion_t, distortion_f = [], []
        for i, (x, y) in enumerate(nodes_list):
            try:
                if self.graph.has_edge(x, y):
                    dlt = self.generator.manifold.sqdist(d_emb[x], d_emb[y])
                    dlf = self.generator.manifold.sqdist(pos_fake_emb[x], d_emb[y])
                    distortion_t.append(abs(self.graph[x][y]["ricciCurvature"] / dlt))
                    distortion_f.append(abs(self.graph[x][y]["ricciCurvature"] / dlf))
            except KeyError:
                continue
        dt, df = torch.cat(distortion_t), torch.cat(distortion_f)
        return (dt - df).norm()
