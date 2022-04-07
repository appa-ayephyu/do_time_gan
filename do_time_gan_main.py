import argparse
import numpy as np
import torch
import torch.utils.data
from do_time_gan_pt.time_gan_pt import time_gan_generator, time_gan_discriminator
from time_gan_pt.models.dataset import TimeGANDataset
from sklearn.model_selection import train_test_split
from do_gan.meta_solvers.prd_solver import projected_replicator_dynamics

# Self-Written Modules
from time_gan_pt.data.data_preprocess import data_preprocess
import tqdm

gan_type = ["time", "image", "spatial"]


class do_gan:
    def __init__(self, args):
        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution
        self.train_max_epoch = args.train_max_epoch
        self.eval_max_epoch = args.eval_max_epoch

        self.data = None
        # self.generator = None
        # self.discriminator = None
        self.generator_list = []
        self.discriminator_list = []

        self.meta_games = [
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.float32),
        ]

        self.meta_strategies = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

    def get_generator(self):
        return time_gan_generator(args=self.args)

    def get_discriminator(self):
        return time_gan_discriminator(args=self.args)

    def get_dataset(self):
        #########################
        # Load and preprocess data for model
        #########################
        args = self.args
        data_path = "time_gan_pt/data/stock.csv"
        X, T, _, args.max_seq_len, args.padding_value = data_preprocess(
            data_path, args.max_seq_len
        )

        print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
        print(f"Original data preview:\n{X[:2, :10, :2]}\n")

        args.feature_dim = X.shape[-1]
        args.Z_dim = X.shape[-1]

        # Train-Test Split data and time
        train_data, test_data, train_time, test_time = train_test_split(
            X, T, test_size=args.train_rate, random_state=args.seed
        )

        dataset = TimeGANDataset(train_data, train_time)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False
        )

        return dataloader

    def init(self):
        self.data = self.get_dataset()
        # Training
        generator = self.get_generator()
        discriminator = self.get_discriminator()

        print("preprocessing of the generator")
        generator.preprocessing_it(self.data)
        gen_loss = 0.0
        dis_loss = 0.0
        sum_idx = 0
        print("evaluating the current generator/discriminator")
        for _ in range(self.eval_max_epoch):
            for X_mb, T_mb in self.data:
                sum_idx += 1
                data = {"X": X_mb, "T": T_mb}
                g_loss = generator.eval_it(data, discriminator)
                gen_loss += g_loss["g_loss"]

                d_loss = discriminator.eval_it(data, generator)
                dis_loss += d_loss["d_loss"]
        gen_loss /= sum_idx
        dis_loss /= sum_idx

        self.generator_list.append(generator)
        self.discriminator_list.append(discriminator)

        r = len(self.generator_list)
        c = len(self.discriminator_list)
        self.meta_games = [
            np.full([r, c], fill_value=-gen_loss),
            np.full([r, c], fill_value=-dis_loss),
        ]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]
        print(self.meta_games)
        print(self.meta_strategies)

    def solve(self):
        dataloader = self.data
        for loop in range(self.max_loop):

            # Training
            generator = self.get_generator()
            discriminator = self.get_discriminator()
            generator.preprocessing_it(
                dataloader=None, pre_generator=self.generator_list[-1]
            )

            logger = tqdm.trange(
                self.args.emb_epochs, desc=f"train the generator {loop}"
            )
            for epoch in logger:
                for X_mb, T_mb in dataloader:
                    dis_idx = np.random.choice(
                        range(len(self.discriminator_list)), p=self.meta_strategies[1]
                    )
                    data = {"X": X_mb, "T": T_mb}
                    generator.train_it(
                        data=data, discriminator=self.discriminator_list[dis_idx]
                    )
            logger = tqdm.trange(
                self.args.emb_epochs, desc=f"train the discriminator {loop}"
            )
            for epoch in logger:
                for X_mb, T_mb in dataloader:
                    gen_idx = np.random.choice(
                        range(len(self.generator_list)), p=self.meta_strategies[0]
                    )
                    data = {"X": X_mb, "T": T_mb}
                    discriminator.train_it(
                        data=data, generator=self.generator_list[gen_idx]
                    )

            # evaluation
            print("augment the game")
            self.generator_list.append(generator)
            self.discriminator_list.append(discriminator)
            r = len(self.generator_list)
            c = len(self.discriminator_list)
            meta_games = [
                np.full([r, c], fill_value=np.nan),
                np.full([r, c], fill_value=np.nan),
            ]
            (o_r, o_c) = self.meta_games[0].shape
            for i in [0, 1]:
                for t_r in range(o_r):
                    for t_c in range(o_c):
                        meta_games[i][t_r][t_c] = self.meta_games[i][t_r][t_c]
            for t_r in range(r):
                for t_c in range(c):
                    if np.isnan(meta_games[0][t_r][t_c]):
                        generator = self.generator_list[t_r]
                        discriminator = self.discriminator_list[t_c]
                        gen_loss = 0.0
                        dis_loss = 0.0
                        sum_idx = 0
                        for _ in range(self.eval_max_epoch):
                            for X_mb, T_mb in self.data:
                                sum_idx += 1
                                data = {"X": X_mb, "T": T_mb}
                                g_loss = generator.eval_it(data, discriminator)
                                gen_loss += g_loss["g_loss"]

                                d_loss = discriminator.eval_it(data, generator)
                                dis_loss += d_loss["d_loss"]
                        gen_loss /= sum_idx
                        dis_loss /= sum_idx
                        meta_games[0][t_r][t_c] = -gen_loss
                        meta_games[1][t_r][t_c] = -dis_loss

            self.meta_games = meta_games
            self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            print(self.meta_games)
            print(self.meta_strategies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_loop", type=int, default=4)
    parser.add_argument(
        "--solution", type=str, default="the solution for the meta game"
    )
    parser.add_argument("--train_max_epoch", type=int, default=10)
    parser.add_argument("--eval_max_epoch", type=int, default=10)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--feat_pred_no", default=2, type=int)

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
    # Data Arguments
    parser.add_argument("--max_seq_len", default=100, type=int)
    parser.add_argument("--train_rate", default=0.5, type=float)

    # Model Arguments
    parser.add_argument("--emb_epochs", default=5, type=int)
    parser.add_argument("--sup_epochs", default=5, type=int)
    parser.add_argument("--gan_epochs", default=600, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=20, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dis_thresh", default=0.05, type=float)  # default 0.15
    parser.add_argument("--optimizer", choices=["adam"], default="adam", type=str)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    args = parser.parse_args()
    print()
    do_time_gan = do_gan(args)
    do_time_gan.init()
    do_time_gan.solve()
