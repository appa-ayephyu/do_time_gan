import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def cost_matrix(x, y, p=2):
    """
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param p: power
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    """
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    b = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    c = torch.sum(b, -1)
    return c


def modified_cost(x, y, h, M):
    """
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L2_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    """
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaMt = M[:, 1:, :] - M[:, :-1, :]
    ht = h[:, :-1, :]
    time_steps = ht.shape[1]
    sum_over_j = torch.sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], -1)
    C_hM = torch.sum(sum_over_j, -1) / time_steps

    # Compute L2 cost $\sum_t^T |x^i_t - y^j_t|^2$
    cost_xy = cost_matrix(x, y)

    return cost_xy + C_hM


def compute_sinkhorn(x, y, h, M, epsilon=0.1, niter=10):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    n = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    C = modified_cost(x, y, h, M)  # shape: [batch_size, batch_size]b

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = 1.0 / n * torch.ones(n, requires_grad=False, device=x.device)
    nu = 1.0 / n * torch.ones(n, requires_grad=False, device=x.device)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -0.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10 ** (-4)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.logsumexp(A, dim=-1, keepdim=True)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0.0 * mu, 0.0 * nu, 0.0
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).item():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def scale_invariante_martingale_regularization(M, reg_lam):
    """
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    """
    m, t, j = M.shape
    # m = torch.tensor(m).type(torch.FloatTensor)
    # t = torch.tensor(m).type(torch.FloatTensor)
    # compute delta M matrix N
    N = M[:, 1:, :] - M[:, :-1, :]
    N_std = N / (torch.std(M, (0, 1)) + 1e-06)

    # Compute \sum_i^m(\delta M)
    sum_m_std = torch.sum(N_std, 0) / m
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = torch.sum(torch.abs(sum_m_std)) / t
    # the total pM term
    pm = reg_lam * sum_across_paths
    return pm


def compute_mixed_sinkhorn_loss(
    f_real,
    f_fake,
    m_real,
    m_fake,
    h_fake,
    sinkhorn_eps,
    sinkhorn_l,
    f_real_p,
    f_fake_p,
    m_real_p,
    h_real_p,
    h_fake_p,
    scale=False,
):
    """
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    """
    f_real = f_real.reshape(f_real.shape[0], f_real.shape[1], -1)
    f_fake = f_fake.reshape(f_fake.shape[0], f_fake.shape[1], -1)
    f_real_p = f_real_p.reshape(f_real_p.shape[0], f_real_p.shape[1], -1)
    f_fake_p = f_fake_p.reshape(f_fake_p.shape[0], f_fake_p.shape[1], -1)
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, sinkhorn_eps, sinkhorn_l)
    loss_xyp = compute_sinkhorn(
        f_real_p, f_fake_p, h_fake_p, m_real_p, sinkhorn_eps, sinkhorn_l
    )
    loss_xx = compute_sinkhorn(
        f_real, f_real_p, h_real_p, m_real, sinkhorn_eps, sinkhorn_l
    )
    loss_yy = compute_sinkhorn(
        f_fake, f_fake_p, h_fake_p, m_fake, sinkhorn_eps, sinkhorn_l
    )

    loss = loss_xy + loss_xyp - loss_xx - loss_yy
    return loss


class VideoDCD(nn.Module):
    """
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    """

    def __init__(
        self,
        batch_size,
        time_steps,
        x_h=64,
        x_w=64,
        filter_size=128,
        j=16,
        nchannel=1,
        bn=False,
    ):
        super(VideoDCD, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.filter_size = filter_size
        self.nchannel = nchannel
        self.ks = 6
        # j is the dimension of h and M
        self.j = j
        self.bn = bn
        self.x_height = x_h
        self.x_width = x_w

        h_in = 8
        s = 2
        p = self.compute_padding(h_in, s, self.ks)

        conv_layers = [
            nn.Conv2d(
                self.nchannel,
                self.filter_size,
                kernel_size=[self.ks, self.ks],
                stride=[s, s],
                padding=[p, p],
            )
        ]
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size))
        conv_layers.append(nn.LeakyReLU())
        conv_layers.append(
            nn.Conv2d(
                self.filter_size,
                self.filter_size * 2,
                kernel_size=[self.ks, self.ks],
                stride=[s, s],
                padding=[p, p],
            )
        )
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size * 2))
        conv_layers.append(nn.LeakyReLU())
        conv_layers.append(
            nn.Conv2d(
                self.filter_size * 2,
                self.filter_size * 4,
                kernel_size=[self.ks, self.ks],
                stride=[s, s],
                padding=[p, p],
            )
        )
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size * 4))
        conv_layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*conv_layers)

        self.lstm1 = nn.LSTM(
            self.filter_size * 4 * 8 * 8, self.filter_size * 4, batch_first=True
        )
        self.lstmbn = nn.BatchNorm1d(self.filter_size * 4)
        self.lstm2 = nn.LSTM(self.filter_size * 4, self.j, batch_first=True)
        # self.sig = nn.Sigmoid()

    # padding computation when h_in = 2h_out
    def compute_padding(self, h_in, s, k_size):
        return max((h_in * (s - 2) - s + k_size) // 2, 0)

    def forward(self, inputs):
        x = inputs.reshape(
            self.batch_size * self.time_steps,
            self.nchannel,
            self.x_height,
            self.x_width,
        )
        x = self.conv_net(x)
        x = x.reshape(self.batch_size, self.time_steps, -1)
        # first output dimension is the sequence of h_t.
        # second output is h_T and c_T(last cell state at t=T).
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        # x = self.sig(x)
        return x


class VideoDCG(nn.Module):
    """
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    """

    def __init__(
        self,
        batch_size=8,
        time_steps=32,
        x_h=64,
        x_w=64,
        filter_size=32,
        state_size=32,
        nchannel=1,
        z_dim=25,
        y_dim=20,
        bn=False,
        output_act="sigmoid",
    ):
        super(VideoDCG, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.filter_size = filter_size
        self.state_size = state_size
        self.nchannel = nchannel
        self.n_noise_t = z_dim
        self.n_noise_y = y_dim
        self.x_height = x_h
        self.x_width = x_w
        self.bn = bn
        self.output_activation = output_act

        self.lstm1 = nn.LSTM(
            self.n_noise_t + self.n_noise_y, self.state_size, batch_first=True
        )
        self.lstmbn1 = nn.BatchNorm1d(self.state_size)
        self.lstm2 = nn.LSTM(self.state_size, self.state_size * 2, batch_first=True)
        self.lstmbn2 = nn.BatchNorm1d(self.state_size * 2)

        dense_layers = [nn.Linear(self.state_size * 2, 8 * 8 * self.filter_size * 4)]
        if self.bn:
            dense_layers.append(nn.BatchNorm1d(8 * 8 * self.filter_size * 4))
        dense_layers.append(nn.LeakyReLU())

        self.dense_net = nn.Sequential(*dense_layers)

        h_in = 8
        s = 2
        k_size = 6
        p = self.compute_padding(h_in, s, k_size)

        deconv_layers = [
            nn.ConvTranspose2d(
                self.filter_size * 4,
                self.filter_size * 4,
                kernel_size=[k_size, k_size],
                stride=[s, s],
                padding=[p, p],
            )
        ]

        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size * 4))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(
            nn.ConvTranspose2d(
                self.filter_size * 4,
                self.filter_size * 2,
                kernel_size=[k_size, k_size],
                stride=[s, s],
                padding=[p, p],
            )
        )

        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size * 2))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(
            nn.ConvTranspose2d(
                self.filter_size * 2,
                self.filter_size,
                kernel_size=[6, 6],
                stride=[2, 2],
                padding=[p, p],
            )
        )
        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(
            nn.ConvTranspose2d(
                self.filter_size,
                self.nchannel,
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
            )
        )

        if self.output_activation == "sigmoid":
            deconv_layers.append(nn.Sigmoid())
        elif self.output_activation == "tanh":
            deconv_layers.append(nn.Tanh())
        else:
            deconv_layers = deconv_layers

        self.deconv_net = nn.Sequential(*deconv_layers)

    # padding computation when 2h_in = h_out
    def compute_padding(self, h_in, s, k_size):
        return max((h_in * (s - 2) - s + k_size) // 2, 0)

    def forward(self, z, y):
        z = z.reshape(self.batch_size, self.time_steps, self.n_noise_t)
        y = y[:, None, :].expand(self.batch_size, self.time_steps, self.n_noise_y)
        x = torch.cat([z, y], -1)
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn2(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(self.batch_size * self.time_steps, -1)
        x = self.dense_net(x)
        x = x.reshape(self.batch_size * self.time_steps, self.filter_size * 4, 8, 8)
        x = self.deconv_net(x)
        x = x.reshape(self.batch_size, self.time_steps, self.x_height, self.x_width)
        return x


class cot_gan_generator(nn.Module):
    def __int__(self, args):
        self.args = args
        self.device = args.device

        # filter size for (de)convolutional layers
        g_state_size = args.g_state_size
        # d_state_size = args.d_state_size
        g_filter_size = args.g_filter_size
        # d_filter_size = args.d_filter_size
        # reg_penalty = args.reg_penalty
        # nlstm = args.n_lstm
        x_width = 64
        x_height = 64
        # channels = args.n_channels
        bn = args.batch_norm
        # Number of RNN layers stacked together
        gen_lr = args.lr
        # disc_lr = args.lr
        time_steps = args.time_steps
        batch_size = args.batch_size

        self.generator = VideoDCG(
            batch_size,
            time_steps,
            x_h=x_height,
            x_w=x_width,
            filter_size=g_filter_size,
            state_size=g_state_size,
            bn=bn,
            output_act="sigmoid",
        ).to(self.device)
        beta1 = 0.5
        beta2 = 0.9
        self.g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=gen_lr, betas=(beta1, beta2)
        )

    def preprocessing_it(self, dataloader):
        pass

    def train_it(self, data, discriminator):
        print("This is for training the model")
        self.generator.train()
        gen_loss = self.forward(data, discriminator)
        # updating Generator
        self.g_opt.zero_grad()
        gen_loss.backward()
        self.g_opt.step()

    def eval_it(
        self,
        data,
        discriminator,
    ):
        print("This is for computing the meta game")
        self.generator.eval()
        gen_loss = self.forward(data, discriminator)

        return {"g_loss": gen_loss.detach().cpu().numpy()}

    def inference_it(self, data):
        print("This is for evaluate the model")

    def forward(self, data, discriminator):
        # print()
        x = data["x"]
        batch_size = self.args.batch_size
        time_steps = self.args.time_steps
        reg_penalty = self.args.reg_penalty
        # it_counts = 0
        sinkhorn_eps = self.args.sinkhorn_eps
        sinkhorn_l = self.args.sinkhorn_l
        z_width = self.args.z_dims_t
        z_height = self.args.z_dims_t
        y_dim = self.args.y_dims

        # Train G
        z = torch.randn(batch_size, time_steps, z_height * z_width)
        y = torch.randn(batch_size, y_dim)
        z_p = torch.randn(batch_size, time_steps, z_height * z_width)
        y_p = torch.randn(batch_size, y_dim)
        real_data = x[:batch_size, ...]
        real_data_p = x[batch_size:, ...]

        fake_data = self.generator(z, y)
        fake_data_p = self.generator(z_p, y_p)

        discriminator_h = discriminator.discriminator_h
        discriminator_m = discriminator.discriminator_m

        h_fake = discriminator_h(fake_data)

        m_real = discriminator_m(real_data)
        m_fake = discriminator_m(fake_data)

        h_real_p = discriminator_h(real_data_p)
        h_fake_p = discriminator_h(fake_data_p)

        m_real_p = discriminator_m(real_data_p)

        real_data = real_data.reshape(batch_size, time_steps, -1)
        fake_data = fake_data.reshape(batch_size, time_steps, -1)
        real_data_p = real_data_p.reshape(batch_size, time_steps, -1)
        fake_data_p = fake_data_p.reshape(batch_size, time_steps, -1)

        gen_loss = compute_mixed_sinkhorn_loss(
            real_data,
            fake_data,
            m_real,
            m_fake,
            h_fake,
            sinkhorn_eps,
            sinkhorn_l,
            real_data_p,
            fake_data_p,
            m_real_p,
            h_real_p,
            h_fake_p,
        )
        return gen_loss


class cot_gan_discriminator(nn.Module):
    def __int__(self, args):
        self.args = args
        self.device = args.device
        d_filter_size = args.d_filter_size
        channels = args.n_channels
        bn = args.batch_norm
        disc_lr = args.lr
        time_steps = args.time_steps
        batch_size = args.batch_size

        self.discriminator_h = VideoDCD(
            batch_size, time_steps, filter_size=d_filter_size, nchannel=channels, bn=bn
        ).to(self.device)
        self.discriminator_m = VideoDCD(
            batch_size, time_steps, filter_size=d_filter_size, nchannel=channels, bn=bn
        ).to(self.device)
        beta1 = 0.5
        beta2 = 0.9
        self.d_opt = torch.optim.Adam(
            params=(
                list(self.discriminator_h.parameters())
                + list(self.discriminator_m.parameters())
            ),
            lr=disc_lr,
            betas=(beta1, beta2),
        )

    def preprocessing_it(self, dataloader):
        pass

    def train_it(self, data, generator):
        print("This is for training the model")
        self.discriminator_h.train()
        self.discriminator_m.train()
        disc_loss = self.forward(data, generator)

        self.d_opt.zero_grad()
        disc_loss.backward()
        self.d_opt.step()

    def eval_it(self, data, generator):
        print("This is for computing the meta game")
        self.discriminator_m.eval()
        self.discriminator_h.eval()
        disc_loss = self.forward(data, generator)
        return {"d_loss": disc_loss.detach().cpu().numpy()}

    def forward(self, data, generator):
        # print()
        batch_size = self.args.batch_size
        time_steps = self.args.time_steps
        reg_penalty = self.args.reg_penalty
        # it_counts = 0
        sinkhorn_eps = self.args.sinkhorn_eps
        sinkhorn_l = self.args.sinkhorn_l
        z_width = self.args.z_dims_t
        z_height = self.args.z_dims_t
        y_dim = self.args.y_dims

        x = data["x"]
        z = torch.randn(batch_size, time_steps, z_height * z_width)
        y = torch.randn(batch_size, y_dim)
        z_p = torch.randn(batch_size, time_steps, z_height * z_width)
        y_p = torch.randn(batch_size, y_dim)
        real_data = x[:batch_size, ...]
        real_data_p = x[batch_size:, ...]

        fake_data = generator(z, y)
        fake_data_p = generator(z_p, y_p)

        h_fake = self.discriminator_h(fake_data)

        m_real = self.discriminator_m(real_data)
        m_fake = self.discriminator_m(fake_data)

        h_real_p = self.discriminator_h(real_data_p)
        h_fake_p = self.discriminator_h(fake_data_p)

        m_real_p = self.discriminator_m(real_data_p)

        real_data = real_data.reshape(batch_size, time_steps, -1)
        fake_data = fake_data.reshape(batch_size, time_steps, -1)
        real_data_p = real_data_p.reshape(batch_size, time_steps, -1)
        fake_data_p = fake_data_p.reshape(batch_size, time_steps, -1)

        loss_d = compute_mixed_sinkhorn_loss(
            real_data,
            fake_data,
            m_real,
            m_fake,
            h_fake,
            sinkhorn_eps,
            sinkhorn_l,
            real_data_p,
            fake_data_p,
            m_real_p,
            h_real_p,
            h_fake_p,
        )

        pm1 = scale_invariante_martingale_regularization(m_real, reg_penalty)
        disc_loss = -loss_d + pm1
        return disc_loss
