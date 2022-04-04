# -*- coding: UTF-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import tqdm


from do_time_gan_pt.utils import soft_update


class EmbeddingNetwork(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN"""

    def __init__(self, args):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Embedder Architecture
        self.emb_rnn = torch.nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.emb_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python
        # /keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F)
            - T: input temporal information (B)
        Returns:
            - H: latent space embeddings (B x S x H)
        """
        # Dynamic RNN input for ignoring paddings
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, lengths=T, batch_first=True, enforce_sorted=False
        )

        # 128 x 100 x 71
        H_o, H_t = self.emb_rnn(X_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        # 128 x 100 x 10
        logits = self.emb_linear(H_o)
        # 128 x 100 x 10
        H = self.emb_sigmoid(logits)
        return H


class RecoveryNetwork(torch.nn.Module):
    """The recovery network (decoder) for TimeGAN"""

    def __init__(self, args):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Recovery Architecture
        self.rec_rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.rec_linear = torch.nn.Linear(self.hidden_dim, self.feature_dim)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/
        # python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.rec_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.rec_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the recovering features from latent space to original space
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - X_tilde: recovered data (B x S x F)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H, lengths=T, batch_first=True, enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.rec_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        # 128 x 100 x 71
        X_tilde = self.rec_linear(H_o)
        return X_tilde


class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN"""

    def __init__(self, args):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Supervisor Architecture
        self.sup_rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers - 1,
            batch_first=True,
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/
        # layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H, lengths=T, batch_first=True, enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.sup_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        # 128 x 100 x 10
        logits = self.sup_linear(H_o)
        # 128 x 100 x 10
        H_hat = self.sup_sigmoid(logits)
        return H_hat


class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN"""

    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Generator Architecture
        self.gen_rnn = torch.nn.GRU(
            input_size=self.Z_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/
        # layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, Z, T):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns:
            - H: embeddings (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z, lengths=T, batch_first=True, enforce_sorted=False
        )

        # 128 x 100 x 71
        H_o, H_t = self.gen_rnn(Z_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        # 128 x 100 x 10
        logits = self.gen_linear(H_o)
        # B x S
        H = self.gen_sigmoid(logits)
        return H


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN"""

    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/
        # legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H, lengths=T, batch_first=True, enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits


class time_gan_generator(nn.Module):
    # print()
    def __init__(self, args):
        super(time_gan_generator, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size

        self.args = args

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        # self.discriminator = DiscriminatorNetwork(args)
        self.e_opt = torch.optim.Adam(self.embedder.parameters(), lr=args.learning_rate)
        self.r_opt = torch.optim.Adam(self.recovery.parameters(), lr=args.learning_rate)
        self.s_opt = torch.optim.Adam(
            self.supervisor.parameters(), lr=args.learning_rate
        )
        self.g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=args.learning_rate
        )

    def preprocessing_it(self, dataloader, pre_generator=None):
        if pre_generator is None:
            logger = tqdm.trange(self.args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
            for _ in logger:
                for X_mb, T_mb in dataloader:
                    # Reset gradients
                    self.e_opt.zero_grad()
                    self.r_opt.zero_grad()
                    # Forward Pass
                    data = {"X": X_mb, "T": T_mb}
                    # time = [args.max_seq_len for _ in range(len(T_mb))]
                    _, E_loss0, E_loss_T0 = self._recovery_forward(data)
                    # loss = np.sqrt(E_loss_T0.item())
                    # Backward Pass
                    E_loss0.backward()
                    # Update model parameters
                    self.e_opt.step()
                    self.r_opt.step()
            logger = tqdm.trange(self.args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
            # pretrain of the supervisor
            for _ in logger:
                for X_mb, T_mb in dataloader:
                    data = {"X": X_mb, "T": T_mb}
                    # Reset gradients
                    self.s_opt.zero_grad()
                    # Forward Pass
                    S_loss = self._supervisor_forward(data)
                    # Backward Pass
                    S_loss.backward()
                    # loss = np.sqrt(S_loss.item())
                    # Update model parameters
                    self.s_opt.step()
        else:
            soft_update(
                online=pre_generator.supervisor, target=self.supervisor, tau=0.0
            )
            soft_update(online=pre_generator.embedder, target=self.embedder, tau=0.0)
            soft_update(online=pre_generator.recovery, target=self.recovery, tau=0.0)

    def train_it(self, data, discriminator, train_step=10):
        print("This is for training the model")
        for _ in range(train_step):
            self.g_opt.zero_grad()
            self.s_opt.zero_grad()
            # Z_mb = torch.rand((self.args.batch_size, self.args.max_seq_len, self.args.Z_dim))
            loss = self.forward(data, discriminator)
            loss.backward()
            # # Update model parameters
            self.g_opt.step()
            self.s_opt.step()

            self.e_opt.zero_grad()
            self.r_opt.zero_grad()
            E_loss, _, E_loss_T0 = self._recovery_forward(data)
            E_loss.backward()
            # Update model parameters
            self.e_opt.step()
            self.r_opt.step()

    def eval_it(
        self,
        data,
        discriminator,
    ):
        print("This is for computing the meta game")
        g_loss = self.forward(data, discriminator)
        E_loss, _, E_loss_T0 = self._recovery_forward(data)

        loss_dict = {
            "g_loss": g_loss.detach().cpu().numpy(),
            "e_loss": E_loss.detach().cpu().numpy(),
        }
        return loss_dict

    def inference_it(self, data):
        print("This is for evaluate the model")
        # Generator Forward Pass
        Z = data["Z"]
        T = data["T"]
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        return X_hat.cpu().detach()

    def _recovery_forward(self, data):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        X = data["X"]
        T = data["T"]

        H = self.embedder(X, T)
        X_tilde = self.recovery(H, T)

        # For Joint training
        H_hat_supervise = self.supervisor(H, T)
        G_loss_S = F.mse_loss(
            H_hat_supervise[:, :-1, :], H[:, 1:, :]
        )  # Teacher forcing next output

        # Reconstruction Loss
        E_loss_T0 = F.mse_loss(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss, E_loss0, E_loss_T0

    def _supervisor_forward(self, data):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        X = data["X"]
        T = data["T"]
        # Supervision Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Supervised loss
        S_loss = F.mse_loss(
            H_hat_supervise[:, :-1, :], H[:, 1:, :]
        )  # Teacher forcing next output
        return S_loss

    def forward(self, data, discriminator, gamma=1.0):
        # print()
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        X = data["X"]
        T = data["T"]
        Z = data["Z"]
        t_discriminator = discriminator.discriminator
        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = t_discriminator(H_hat, T)  # Output of supervisor
        Y_fake_e = t_discriminator(E_hat, T)  # Output of generator

        G_loss_U = F.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = F.binary_cross_entropy_with_logits(
            Y_fake_e, torch.ones_like(Y_fake_e)
        )

        # 2. Supervised loss
        G_loss_S = F.mse_loss(
            H_hat_supervise[:, :-1, :], H[:, 1:, :]
        )  # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(
            torch.abs(
                torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
            )
        )
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = (
            G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
        )

        return G_loss


class time_gan_discriminator(nn.Module):
    def __init__(self, args):
        super(time_gan_discriminator, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size

        self.args = args

        # self.embedder = EmbeddingNetwork(args)
        # self.recovery = RecoveryNetwork(args)
        # self.generator = GeneratorNetwork(args)
        # self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=args.learning_rate
        )

    def preprocessing_it(self):
        # print()
        pass

    def train_it(self, data, generator):
        # print()
        self.discriminator.train()
        # Discriminator Training
        self.d_opt.zero_grad()
        # Forward Pass
        # D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")
        D_loss = self.forward(data, generator)
        # Check Discriminator loss
        if D_loss > self.args.dis_thresh:
            # Backward Pass
            D_loss.backward()

            # Update model parameters
            self.d_opt.step()
        # D_loss = D_loss.item()

    def eval_it(self, data, generator):
        # print()
        self.discriminator.eval()
        d_loss = self.forward(data, generator)
        return {"d_loss": d_loss.detach().cpu().numpy()}

    def forward(self, data, generator, gamma=1.0):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        X = data["X"]
        T = data["T"]
        Z = data["Z"]

        embedder = generator.embedder
        supervisor = generator.supervisor
        t_generator = generator.generator
        H = embedder(X, T).detach()

        # Generator
        E_hat = t_generator(Z, T).detach()
        H_hat = supervisor(E_hat, T).detach()

        # Forward Pass
        Y_real = self.discriminator(H, T)  # Encoded original data
        Y_fake = self.discriminator(H_hat, T)  # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat, T)  # Output of generator

        D_loss_real = F.binary_cross_entropy_with_logits(
            Y_real, torch.ones_like(Y_real)
        )
        D_loss_fake = F.binary_cross_entropy_with_logits(
            Y_fake, torch.zeros_like(Y_fake)
        )
        D_loss_fake_e = F.binary_cross_entropy_with_logits(
            Y_fake_e, torch.zeros_like(Y_fake_e)
        )

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss
