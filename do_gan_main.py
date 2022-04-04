import argparse
import numpy as np
import torch
import torch.utils.data
from do_time_gan_pt.time_gan_pt import time_gan_generator, time_gan_discriminator
from time_gan_pt.models.dataset import TimeGANDataset
from sklearn.model_selection import train_test_split

# Self-Written Modules
from time_gan_pt.data.data_preprocess import data_preprocess

gan_type = ["time", "image", "spatial"]


class do_gan:
    def __init__(self, args):
        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution

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

    def solve(self):
        for _ in range(self.max_loop):
            generator = self.get_generator()
            discriminator = self.get_discriminator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_loop", type=int, default=10)
    parser.add_argument(
        "--solution", type=str, default="the solution for the meta game"
    )
    args = parser.parse_args()
    print()
