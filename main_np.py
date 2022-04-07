# # # import numpy as np
# # # import argparse
# # # new_meta_payoff_matrix = np.full([5, 5], fill_value=np.nan)
# # #
# # # print(new_meta_payoff_matrix)
# # #
# # # print(new_meta_payoff_matrix.shape)
# # # (row, col) = new_meta_payoff_matrix.shape
# # # print(row)
# # # print(col)
# # #
# # # meta = np.empty([5, 5])
# # # print(meta)
# # #
# # # from meta_solvers.prd_solver import projected_replicator_dynamics
# # #
# # # meta_games = [np.array([[3, 0], [4, 1]]), np.array([[3, 4], [0, 1]])]
# # # print(meta_games)
# # #
# # # res = projected_replicator_dynamics(meta_games)
# # # print(res)
# #
# # import copy
# #
# # # list_a = [2, 3, 4]
# # #
# # # list_b = deepcopy(list_a)
# # #
# # # list_a[0] = 4
# # #
# # # print(list_a)
# # # print(list_b)
# #
# # from time_gan_pt.models.timegan import EmbeddingNetwork
# #
# # import argparse
# #
# # # Inputs for the main function
# # parser = argparse.ArgumentParser()
# #
# # # Experiment Arguments
# # parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
# # parser.add_argument("--exp", default="test", type=str)
# # # parser.add_argument("--is_train", type=str2bool, default=True)
# # parser.add_argument("--seed", default=0, type=int)
# # parser.add_argument("--feat_pred_no", default=2, type=int)
# #
# # # Data Arguments
# # parser.add_argument("--max_seq_len", default=100, type=int)
# # parser.add_argument("--train_rate", default=0.5, type=float)
# #
# # # Model Arguments
# # parser.add_argument("--emb_epochs", default=20, type=int)
# # parser.add_argument("--sup_epochs", default=20, type=int)
# # parser.add_argument("--gan_epochs", default=600, type=int)
# # parser.add_argument("--batch_size", default=128, type=int)
# # parser.add_argument("--hidden_dim", default=20, type=int)
# # parser.add_argument("--num_layers", default=3, type=int)
# # parser.add_argument("--dis_thresh", default=0.15, type=float)
# # parser.add_argument("--optimizer", choices=["adam"], default="adam", type=str)
# # parser.add_argument("--learning_rate", default=1e-3, type=float)
# #
# # args = parser.parse_args()
# #
# # args.feature_dim = 10
# # args.Z_dim = 20
# # args.padding_value = 1
# #
# # emb_net_1 = EmbeddingNetwork(args)
# #
# # print(emb_net_1.emb_linear.parameters())
# # for para in emb_net_1.emb_linear.parameters():
# #     print(para)
# # emb_net_2 = copy.deepcopy(emb_net_1)
# # print(id(emb_net_1.emb_linear))
# # print(id(emb_net_2.emb_linear))
# # # for para in emb_net_2.parameters():
# # #     para.data = para.data * 0.5
# # # print(emb_net_2.emb_linear.parameters())
# # #
# # # for para in emb_net_1.emb_linear.parameters():
# # #     print(para)
# # #
# # # for i in range(5):
# # #     print()
# # #     print()
# # # for para in emb_net_2.emb_linear.parameters():
# # #     print(para)
# # #
# # # for i in range(5):
# # #     print()
# # #     print()
# # # for para in emb_net_1.emb_linear.parameters():
# # #     print(para)
# # # print(id(emb_net_1.emb_linear.parameters()))
# # # print(id(emb_net_2.emb_linear.parameters()))
# # #
# # # print(id(emb_net_1.emb_linear.parameters()) == id(emb_net_2.emb_linear.parameters()))
# # import torch
# # import torch.nn as nn
# # import numpy as np
# # node_emd_init = np.random.random([10, 10])
# # embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
# # print(embedding_matrix)
# #
# # bias = nn.Parameter(torch.tensor(node_emd_init))
# # g_opt = torch.optim.Adam([embedding_matrix, bias], lr=0.001)
# # print(g_opt)
#
# import numpy as np
#
# meta_game = [np.array([[]]), np.array([[]])]
#
# print(meta_game)
#
# for game in meta_game:
#     print(game.shape)
#     # print(r)
#     # print(c)
#
#
# def empty_list_generator(number_dimensions):
#     result = []
#     for _ in range(number_dimensions - 1):
#         result = [result]
#     return result
#
#
# empty_list = empty_list_generator(number_dimensions=2)
#
# print(empty_list)
import yaml
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_loop", type=int, default=4)
parser.add_argument("--min_loop", type=int, default=4)
args = parser.parse_args()
print(args.__dict__.keys())
args.__dict__['t'] = 10
print(args)
# print(parser)

f = open("do_gan/configs/time_gan.yaml")
# y = yaml.load(f, Loader=yaml.FullLoader)
yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
print(yaml_cfg)
for super_cfg_name, attr_value in yaml_cfg.items():
    for attr, value in attr_value.items():
        args.__dict__[attr] = value
print(args)