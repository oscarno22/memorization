# -*- coding: utf-8 -*-

"""
Model definitions

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import abc
import math

import torch
import torch.nn as nn

from typing import Tuple

from torch.nn.functional import binary_cross_entropy
from torch.nn.functional import softplus
from torch.nn.functional import log_softmax

from constants import LOGIT_LAMBDA


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self._device = "cpu"

    @property
    def device(self):
        return self._device

    def to(self, device):
        super().to(device)
        self._device = device

    @abc.abstractmethod
    def step(self, batch: torch.Tensor):
        pass

    @abc.abstractmethod
    def loss_function(*args, **kwargs):
        pass


class BaseVAE(BaseModel):
    @property
    def latent_dim(self):
        return self._latent_dim

    @abc.abstractproperty
    def description(self):
        """Description of the model to store in the output files"""

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x is expected to be of shape (batch_size, n_channel, height, width)"""
        xx = self._encoder(x)
        return self._mu(xx), self._logvar(xx)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z is expected to be of shape (batch_size, latent_dim)"""
        return self._decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @abc.abstractmethod
    def construct_model(self) -> None:
        """Create the encoder/decoder networks"""

    @abc.abstractmethod
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        """Take a random sample from the decoder given the latent variable"""

    @abc.abstractmethod
    def log_pxz(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute log p(x | z) where pred = decoder(z)"""
        # Inputs assumed to be of shape (B, C, H, W)
        # Output should be of shape (B,)

    def push(self, z: torch.Tensor) -> torch.Tensor:
        """Push a batch of latent vectors through the network"""
        return self.reconstruct(self.decode(z))

    def reconstruct(self, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct the output of the decoder if necessary"""
        return y

    def loss_function(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        logpxz = self.log_pxz(pred, true)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        REC = -logpxz.sum()
        return REC + KLD

    def step(self, batch: torch.Tensor) -> torch.Tensor:
        """Run a single step of the model and return the loss"""
        x_pred, mu, logvar = self(batch)
        loss = self.loss_function(batch, x_pred, mu, logvar)
        return loss


class BernoulliMixin:
    def log_pxz(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Log p(x | z) for Bernoulli decoder"""
        BCE = -binary_cross_entropy(pred, true, reduction="none")
        BCE = BCE.sum(axis=(1, 2, 3))
        return BCE

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decode(z)
        return y.bernoulli_()


class DiagonalGaussianMixin:
    def loss_function(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        logpxz = self.log_pxz_logit(pred, true)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        REC = -logpxz.sum()
        return REC + KLD

    def log_pxz(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Log p(x | z) for Gaussian decoder with constant diagonal cov"""
        D = self._in_channels * self._img_dim * self._img_dim
        logpx_logit = self.log_pxz_logit(pred, true)
        logpx_pixel = (
            logpx_logit
            + D * torch.log(1 - 2 * torch.tensor(LOGIT_LAMBDA))
            - D * torch.log(torch.tensor(256))
            - torch.sum(true - 2 * softplus(true), axis=(1, 2, 3))
        )
        return logpx_pixel

    def log_pxz_logit(self, Y: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Log p(x | z) for Gaussian decoder with diagonal cov. (logit space)"""
        # Both the data and the model pretend that they're in logit space
        # To get bits per dim, see eq (27) of
        # https://arxiv.org/pdf/1705.07057.pdf and use log_pxz output
        C = self._in_channels
        D = self._in_channels * self._img_dim * self._img_dim
        assert Y.shape[1] == 2 * C
        mu_theta = Y[:, :C, :, :]
        logvar_theta = Y[:, C:, :, :]
        logvar_theta = torch.clamp(logvar_theta, min=-7.0)
        inv_std = torch.exp(-0.5 * logvar_theta)
        SSE = torch.sum(torch.square(inv_std * (true - mu_theta)), axis=(1, 2, 3))
        out = -0.5 * (
            D * math.log(2 * math.pi) + logvar_theta.sum(axis=(1, 2, 3)) + SSE
        )
        return out

    def reconstruct(self, y: torch.Tensor) -> torch.Tensor:
        C = self._in_channels
        return y[:, :C, :, :]

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        C = self._in_channels
        y = self.decode(z)
        mu = y[:, :C, :, :]
        logvar = y[:, C:, :, :]
        logvar = torch.clamp(logvar, min=-7)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)


class ConstantGaussianMixin:
    def loss_function(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        logpxz = self.log_pxz_logit(pred, true)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        REC = -logpxz.sum()
        return REC + KLD

    def log_pxz(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Log p(x | z) for Gaussian decoder with constant diagonal cov"""
        D = self._in_channels * self._img_dim * self._img_dim
        logpx_logit = self.log_pxz_logit(pred, true)
        logpx_pixel = (
            logpx_logit
            + D * torch.log(1 - 2 * torch.tensor(LOGIT_LAMBDA))
            - D * torch.log(torch.tensor(256))
            - torch.sum(true - 2 * softplus(true), axis=(1, 2, 3))
        )
        return logpx_pixel

    def log_pxz_logit(self, Y: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Log p(x | z) for Gaussian decoder with learned constant diagonal
        cov.  (logit space)"""
        # Both the data and the model pretend that they're in logit space
        # To get bits per dim, see eq (27) of
        # https://arxiv.org/pdf/1705.07057.pdf and use log_pxz output
        D = self._in_channels * self._img_dim * self._img_dim
        mu_theta = Y
        inv_gamma = torch.exp(-self.loggamma)
        SSE = torch.sum(inv_gamma * torch.square(true - mu_theta), axis=(1, 2, 3))
        out = -0.5 * (D * math.log(2 * math.pi) + D * self.loggamma + SSE)
        return out

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decode(z)
        std = torch.exp(0.5 * self.loggamma)
        return y + std * torch.randn_like(y)


class MixLogisticsMixin:
    def log_pxz(self, pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Log p(x | z) for mixture of discrete logistics decoder"""
        # pred is the output of the decoder
        # This implementation is based on the original PixelCNN++ code and the
        # NVAE code, as well as various online sources on this topic.

        # -- Input validation --
        K = self._num_mixture
        B, M, H_, W_ = pred.shape
        B, C, H, W = X.shape
        assert C == 3 and H == H_ and W == W_ and M == 10 * K
        assert X.min() >= 0.0 and X.max() <= 1.0

        # -- Extract decoder output --
        # pred has 10*K channels: log mixture components (K), means (3*K),
        # log scales (3*K), alpha (K), beta (K), and gamma (K).
        log_mixture_comp = pred[:, :K, :, :]  #         B, K, H, W
        means = pred[:, K : 4 * K, :, :]  #             B, 3K, H, W
        log_scales = pred[:, 4 * K : 7 * K, :, :]  #    B, 3K, H, W
        alpha = pred[:, 7 * K : 8 * K, :, :]  #         B, K, H, W
        beta = pred[:, 8 * K : 9 * K, :, :]  #          B, K, H, W
        gamma = pred[:, 9 * K : 10 * K, :, :]  #        B, K, H, W

        # -- Clipping and mapping --
        # Map the X values to [-1, 1]
        X = 2 * X - 1

        # Clamp log scales to avoid exploding scale values
        log_scales = log_scales.clamp(min=-7.0)

        # Keep coefficients between -1 and +1
        alpha = torch.tanh(alpha)
        beta = torch.tanh(beta)
        gamma = torch.tanh(gamma)

        # -- Reconfigure for easier computation --

        # Replicate X into another dimension
        X = X.unsqueeze(4)  #                          B, C, H, W, 1
        X = X.expand(-1, -1, -1, -1, K)  #             B, C, H, W, K
        X = X.permute(0, 1, 4, 2, 3)  #                B, C, K, H, W

        # Reshape the means and logscales to match
        means = means.view(B, C, K, H, W)
        log_scales = log_scales.view(B, C, K, H, W)

        # -- Compute the means for the different subpixels --
        mean_r = means[:, 0, :, :, :]  #                B, 1, K, H, W
        mean_g = means[:, 1, :, :, :] + alpha * X[:, 0, :, :, :]
        mean_b = (
            means[:, 2, :, :, :] + beta * X[:, 0, :, :, :] + gamma * X[:, 1, :, :, :]
        )

        # Combine means
        mean_r = mean_r.unsqueeze(1)
        mean_g = mean_g.unsqueeze(1)
        mean_b = mean_b.unsqueeze(1)
        means = torch.cat([mean_r, mean_g, mean_b], axis=1)  # B, C, K, H, W

        # Compute x - mu for each channel and mixture component
        centered = X - means  # B, C, K, H, W

        # Compute inverse scale of logistics
        inv_scale = torch.exp(-log_scales)

        # Compute U_plus = (x + 1/2 - mu)/s and U_min = (x - 1/2 - mu)/s.
        # Because x is in [-1, 1] instead of [0, 255], 1/2 becomes 1/255.
        U_plus = inv_scale * (centered + 1.0 / 255.0)
        U_min = inv_scale * (centered - 1.0 / 255.0)

        # Apply sigmoid and compute difference (for non edge-case)
        cdf_plus = torch.sigmoid(U_plus)  # B, C, K, H, W
        cdf_min = torch.sigmoid(U_min)  #   B, C, K, H, W
        cdf_delta = cdf_plus - cdf_min  #   B, C, K, H, W

        # -- Compute values for edge cases --
        # For x = 0
        log_cdf_plus = U_plus - softplus(U_plus)

        # For x = 255
        log_one_minus_cdf_min = -softplus(U_min)

        # Midpoint fix. When cdf_delta is very small (here, smaller than 1e-5),
        # the difference in CDF values is small. Thus, the small difference in
        # CDF can be approximated by a derivative. Recall that for a CDF F(x)
        # and a PDF f(x) we have lim_{t -> 0} (F(x + t) - F(x - t))/(2*t) =
        # f(x). So here the PixelCNN++ authors approximate F(x + t) - F(x - t)
        # by 2*t*f(x). And since we take logs and t = 1/255, we get log(2 *
        # 1/255) = log(127.5) and log f(x). This gives for log f(x) (pdf of
        # logistic distribution):
        U_mid = inv_scale * centered
        log_pdf_mid = U_mid - log_scales - 2.0 * softplus(U_mid)

        # -- Combine log probabilitie --

        # Compute safe (non-weighted) log prob for non edge-cases
        # Note that clamp on log(cdf_delta) is needed for backprop (nan can
        # occur)
        log_prob_mid = torch.where(
            cdf_delta > 1e-5,
            torch.log(torch.clamp(cdf_delta, min=1e-10)),
            log_pdf_mid - torch.log(torch.tensor(255.0 / 2)),
        )

        # Determine boundaries for edge cases
        # NOTE: This differs slightly from other implementations, but
        # corresponds to the theoretical values.
        left = 0.5 / 255 * 2 - 1  # right boundary for x=0 on [-1, 1]
        right = (255 - 0.5) / 255 * 2 - 1  # left boundary for x=255 on [-1, 1]

        # Compute (non-weighted) log prob for all cases
        log_prob = torch.where(
            X < left,
            log_cdf_plus,
            torch.where(X > right, log_one_minus_cdf_min, log_prob_mid),
        )

        # Sum over channels (channel probs are multiplied, so log probs sum)
        # and weight with mixture component weights (in log space, so we use
        # log_softmax to ensure mixture_comp sums to 1).
        log_prob = log_prob.sum(axis=1) + log_softmax(log_mixture_comp, dim=1)

        # log prob is (B, K, H, W), so we logsumexp over everything but B
        return torch.logsumexp(log_prob, dim=(1, 2, 3))


class BernoulliMLPVAE(BernoulliMixin, BaseVAE):
    _layers = [512, 256]

    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 2,
        in_channels: int = 1,
        **kwargs,
    ):
        super().__init__()
        self._img_dim = img_dim
        self._latent_dim = latent_dim
        self._in_channels = in_channels
        self.construct_model()

    @property
    def description(self):
        layers = "-".join(map(str, self._layers))
        latent = str(self._latent_dim)
        d = f"{self.__class__.__name__}_{layers}-{latent}"
        return d

    def construct_model(self):
        C = self._in_channels
        D = self._img_dim
        L = self._latent_dim

        input_shape = D * D * C

        encoder = [nn.Flatten()]
        prev_dim = input_shape
        for l in self._layers:
            encoder.append(nn.Linear(prev_dim, l))
            encoder.append(nn.ReLU(True))
            prev_dim = l

        self._encoder = nn.Sequential(*encoder)
        self._mu = nn.Linear(prev_dim, L)
        self._logvar = nn.Linear(prev_dim, L)

        decoder = []
        prev_dim = L
        for l in reversed(self._layers):
            decoder.append(nn.Linear(prev_dim, l))
            decoder.append(nn.ReLU(True))
            prev_dim = l
        decoder.append(nn.Linear(prev_dim, input_shape))
        decoder.append(nn.Sigmoid())
        decoder.append(nn.Unflatten(1, (C, D, D)))
        self._decoder = nn.Sequential(*decoder)


class BernoulliDCVAE(BernoulliMixin, BaseVAE):
    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 2,
        in_channels: int = 1,
        num_feature: int = 64,
    ):
        super().__init__()
        self._img_dim = img_dim
        self._latent_dim = latent_dim
        self._in_channels = in_channels
        self._num_feature = num_feature
        self.construct_model()

    @property
    def description(self):
        d = f"{self.__class__.__name__}_NF{self._num_feature}-L{self._latent_dim}"
        return d

    def construct_model(self):
        C = self._in_channels
        D = self._img_dim
        L = self._latent_dim
        F = self._num_feature

        # Model is designed for 32x32 input/output
        assert D == 32

        self._encoder = nn.Sequential(
            nn.Conv2d(C, F, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F) x 16 x 16
            nn.Conv2d(F, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*2) x 8 x 8
            nn.Conv2d(F * 2, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*4) x 4 x 4
            nn.Conv2d(F * 4, F * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (F*8) x 2 x 2
            # Flatten layer
            nn.Flatten(),
        )

        prev_dim = int((F * 8) * 2 * 2)
        self._mu = nn.Linear(prev_dim, L)
        self._logvar = nn.Linear(prev_dim, L)

        self._decoder = nn.Sequential(
            # input is Z, going into a convolution
            # NOTE: Using kernel_size = 2 here to get 32x32 output
            nn.Unflatten(1, (L, 1, 1)),
            nn.ConvTranspose2d(L, F * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.ReLU(True),
            # state size. (F*8) x 2 x 2
            nn.ConvTranspose2d(F * 8, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.ReLU(True),
            # state size. (F*4) x 4 x 4
            nn.ConvTranspose2d(F * 4, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.ReLU(True),
            # state size. (F*2) x 8 x 8
            nn.ConvTranspose2d(F * 2, F, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(True),
            # state size. (F) x 16 x 16
            nn.ConvTranspose2d(F, C, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (C) x 32 x 32
        )


class MixLogisticsDCVAE(MixLogisticsMixin, BaseVAE):
    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 2,
        in_channels: int = 3,
        num_feature: int = 32,
        num_mixture: int = 5,
    ):
        super().__init__()
        if not in_channels == 3:
            raise NotImplementedError
        self._img_dim = img_dim
        self._latent_dim = latent_dim
        self._in_channels = in_channels
        self._num_feature = num_feature
        self._num_mixture = num_mixture
        self.construct_model()

    @property
    def description(self):
        d = f"{self.__class__.__name__}_NF{self._num_feature}-L{self._latent_dim}"
        return d

    def construct_model(self):
        K = self._num_mixture
        C = self._in_channels
        D = self._img_dim
        L = self._latent_dim
        F = self._num_feature

        # Model is designed for 32x32 input/output
        assert D == 32

        self._encoder = nn.Sequential(
            nn.Conv2d(C, F, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F) x 16 x 16
            nn.Conv2d(F, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*2) x 8 x 8
            nn.Conv2d(F * 2, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*4) x 4 x 4
            nn.Conv2d(F * 4, F * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (F*8) x 2 x 2
            # Flatten layer
            nn.Flatten(),
        )

        prev_dim = int((F * 8) * 2 * 2)
        self._mu = nn.Linear(prev_dim, L)
        self._logvar = nn.Linear(prev_dim, L)

        self._decoder = nn.Sequential(
            # input is Z, going into a convolution
            # NOTE: Using kernel_size = 2 here to get 32x32 output
            nn.Unflatten(1, (L, 1, 1)),
            nn.ConvTranspose2d(L, F * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.ReLU(True),
            # state size. (F*8) x 2 x 2
            nn.ConvTranspose2d(F * 8, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.ReLU(True),
            # state size. (F*4) x 4 x 4
            nn.ConvTranspose2d(F * 4, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.ReLU(True),
            # state size. (F*2) x 8 x 8
            nn.ConvTranspose2d(F * 2, F, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(True),
            # state size. (F) x 16 x 16
            nn.ConvTranspose2d(F, 10 * K, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (10K) x 32 x 32
        )


class DiagonalGaussianDCVAE(DiagonalGaussianMixin, BaseVAE):
    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 2,
        in_channels: int = 3,
        num_feature: int = 64,
    ):
        super().__init__()
        self._img_dim = img_dim
        self._latent_dim = latent_dim
        self._in_channels = in_channels
        self._num_feature = num_feature
        self.construct_model()

    @property
    def description(self):
        d = f"{self.__class__.__name__}_NF{self._num_feature}-L{self._latent_dim}"
        return d

    def construct_model(self):
        C = self._in_channels
        F = self._num_feature
        L = self._latent_dim

        self._encoder = nn.Sequential(
            nn.Conv2d(C, F, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F) x 16 x 16
            nn.Conv2d(F, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*2) x 8 x 8
            nn.Conv2d(F * 2, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*4) x 4 x 4
            nn.Conv2d(F * 4, F * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (F*8) x 2 x 2
            # Flatten layer
            nn.Flatten(),
        )

        prev_dim = int((F * 8) * 2 * 2)
        self._mu = nn.Linear(prev_dim, L)
        self._logvar = nn.Linear(prev_dim, L)

        self._decoder = nn.Sequential(
            # input is Z, going into a convolution
            # NOTE: Using kernel_size = 2 here to get 32x32 output
            nn.Unflatten(1, (L, 1, 1)),
            nn.ConvTranspose2d(L, F * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.ReLU(True),
            # state size. (F*8) x 2 x 2
            nn.ConvTranspose2d(F * 8, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.ReLU(True),
            # state size. (F*4) x 4 x 4
            nn.ConvTranspose2d(F * 4, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.ReLU(True),
            # state size. (F*2) x 8 x 8
            nn.ConvTranspose2d(F * 2, F, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(True),
            # state size. (F) x 16 x 16
            nn.ConvTranspose2d(F, 2 * C, 4, 2, 1, bias=False),
            # state size. (2C) x 32 x 32
        )


class ConstantGaussianDCVAE(ConstantGaussianMixin, BaseVAE):
    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 2,
        in_channels: int = 3,
        num_feature: int = 64,
    ):
        super().__init__()
        self._img_dim = img_dim
        self._latent_dim = latent_dim
        self._in_channels = in_channels
        self._num_feature = num_feature
        self.construct_model()

    @property
    def description(self):
        d = f"{self.__class__.__name__}_NF{self._num_feature}-L{self._latent_dim}"
        return d

    def construct_model(self):
        C = self._in_channels
        F = self._num_feature
        L = self._latent_dim

        self.loggamma = nn.Parameter(torch.tensor(-2.0))

        self._encoder = nn.Sequential(
            nn.Conv2d(C, F, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F) x 16 x 16
            nn.Conv2d(F, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*2) x 8 x 8
            nn.Conv2d(F * 2, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (F*4) x 4 x 4
            nn.Conv2d(F * 4, F * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (F*8) x 2 x 2
            # Flatten layer
            nn.Flatten(),
        )

        prev_dim = int((F * 8) * 2 * 2)
        self._mu = nn.Linear(prev_dim, L)
        self._logvar = nn.Linear(prev_dim, L)

        self._decoder = nn.Sequential(
            # input is Z, going into a convolution
            # NOTE: Using kernel_size = 2 here to get 32x32 output
            nn.Unflatten(1, (L, 1, 1)),
            nn.ConvTranspose2d(L, F * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.ReLU(True),
            # state size. (F*8) x 2 x 2
            nn.ConvTranspose2d(F * 8, F * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 4),
            nn.ReLU(True),
            # state size. (F*4) x 4 x 4
            nn.ConvTranspose2d(F * 4, F * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F * 2),
            nn.ReLU(True),
            # state size. (F*2) x 8 x 8
            nn.ConvTranspose2d(F * 2, F, 4, 2, 1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(True),
            # state size. (F) x 16 x 16
            nn.ConvTranspose2d(F, C, 4, 2, 1, bias=False),
            # state size. (2C) x 32 x 32
        )


class ActNorm1d(nn.Module):
    """Activation Normalization (per-feature) with log-det.
    Initialized on first batch to have zero mean and unit variance.
    After that parameters are fixed (learnable like in Glow variant).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("_initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x: torch.Tensor):
        # x: (B, D)
        with torch.no_grad():
            mean = x.mean(0)
            std = x.std(0) + self.eps
            self.bias.data.copy_(mean)
            self.log_scale.data.copy_(torch.log(1.0 / std))
            self._initialized.data.fill_(1)

    def forward(self, x: torch.Tensor):
        if self._initialized.item() == 0:
            self.initialize(x)
        y = (x - self.bias) * torch.exp(self.log_scale)
        log_det = torch.sum(self.log_scale) * torch.ones(x.size(0), device=x.device)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        x = y * torch.exp(-self.log_scale) + self.bias
        log_det = -torch.sum(self.log_scale) * torch.ones(y.size(0), device=y.device)
        return x, log_det


class SimpleRealNVP(BaseModel):
    """
    Lightweight Real NVP normalizing flow model for image data (flattened).
    Differences vs previous version:
    - Removes BatchNorm1d (previously missing log-det contribution)
    - Adds ActNorm1d layers with proper log-det accounting
    - Centralizes forward/inverse transforms (flow_forward / flow_inverse)
    - Correct log_prob computation (base log prob + total log-det)
    NOTE: description kept stable to satisfy Makefile target patterns.
    """

    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = None,  # compatibility argument (unused)
        in_channels: int = 1,
        num_layers: int = 4,
        hidden_dim: int = 256,
        num_feature: int = None,  # mapped to hidden_dim if provided via loader
        **kwargs,
    ):
        super().__init__()
        self._img_dim = img_dim
        self._in_channels = in_channels
        if num_feature is not None:
            hidden_dim = num_feature  # allow reuse of --num-feature flag
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._data_dim = img_dim * img_dim * in_channels
        self.construct_model()

    @property
    def description(self):
        # Keep original naming convention expected by Makefile
        return f"{self.__class__.__name__}_L{self._num_layers}-H{self._hidden_dim}"

    @property
    def latent_dim(self):
        return self._data_dim  # flows: latent dimension equals data dimension

    # ---------------- Flow construction ----------------
    def construct_model(self):
        self.masks = []
        self.coupling_nets = nn.ModuleList()
        self.actnorms = nn.ModuleList()
        for i in range(self._num_layers):
            mask = self._create_mask(i % 2)
            self.register_buffer(f"mask_{i}", mask)
            self.masks.append(mask)
            self.coupling_nets.append(self._create_coupling_network())
            self.actnorms.append(ActNorm1d(self._data_dim))

    def _create_mask(self, reverse: bool = False):
        mask = torch.zeros(self._data_dim)
        if reverse:
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask

    def _create_coupling_network(self):
        return nn.Sequential(
            nn.Linear(self._data_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._data_dim * 2),  # outputs s and t
        )

    # --------------- Forward / Inverse transforms ---------------
    def _coupling_forward(self, x, net, mask):
        masked_x = x * mask
        st = net(masked_x)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)  # stabilize
        inv_mask = 1 - mask
        y = masked_x + inv_mask * (x * torch.exp(s) + t)
        log_det = torch.sum(inv_mask * s, dim=1)
        return y, log_det

    def _coupling_inverse(self, y, net, mask):
        masked_y = y * mask
        st = net(masked_y)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        inv_mask = 1 - mask
        # invert: x_unmasked = (y_unmasked - t)*exp(-s)
        y_unmasked = y * inv_mask
        x_unmasked = (y_unmasked - t) * torch.exp(-s)
        x = masked_y + inv_mask * x_unmasked
        log_det = -torch.sum(inv_mask * s, dim=1)
        return x, log_det

    def flow_forward(self, x: torch.Tensor):
        B = x.size(0)
        z = x.view(B, -1)
        total_log_det = torch.zeros(B, device=x.device)
        for i, (net, act) in enumerate(zip(self.coupling_nets, self.actnorms)):
            mask = getattr(self, f"mask_{i}")
            z, log_det_c = self._coupling_forward(z, net, mask)
            z, log_det_a = act(z)
            total_log_det += log_det_c + log_det_a
        return z, total_log_det

    def flow_inverse(self, z: torch.Tensor):
        B = z.size(0)
        x = z
        total_log_det = torch.zeros(B, device=z.device)
        for i in reversed(range(self._num_layers)):
            act = self.actnorms[i]
            net = self.coupling_nets[i]
            mask = getattr(self, f"mask_{i}")
            x, log_det_a = act.inverse(x)
            x, log_det_c = self._coupling_inverse(x, net, mask)
            total_log_det += log_det_a + log_det_c
        return x.view(B, self._in_channels, self._img_dim, self._img_dim), total_log_det

    # --------------- Public API (compatibility) ---------------
    def forward(self, x):
        # For trainer compatibility (returns dummy mu/logvar like VAE)
        z, _ = self.flow_forward(x)
        return x, torch.zeros_like(z), torch.zeros_like(z)

    def encode(self, x):  # not meaningful for flow
        B = x.size(0)
        dummy_mu = torch.zeros(B, self._data_dim, device=x.device)
        dummy_logvar = torch.zeros(B, self._data_dim, device=x.device)
        return dummy_mu, dummy_logvar

    def decode(self, z):  # inverse transform for compatibility
        x, _ = self.flow_inverse(z.view(-1, self._data_dim))
        return x

    def log_prob(self, x):
        z, log_det = self.flow_forward(x)
        log_pz = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self._data_dim * math.log(
            2 * math.pi
        )
        return log_pz + log_det

    def log_pxz(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # For flow: full log likelihood
        return self.log_prob(true)

    def sample(self, num_samples: int):
        with torch.no_grad():
            z = torch.randn(num_samples, self._data_dim, device=self.device)
            x, _ = self.flow_inverse(z)
            return x

    def step(self, batch: torch.Tensor):
        return -self.log_prob(batch).mean()

    def loss_function(self, *args, **kwargs):
        return self.step(args[0])

    def marginal_batch_flow(self, batch: torch.Tensor, N: int = 256) -> torch.Tensor:
        return self.log_prob(batch)


def load_model(
    model_type: str,
    img_dim: int,
    latent_dim: int = 32,
    in_channels: int = 1,
    num_feature: int = 32,
) -> BaseModel:
    model_cls = MODELS.get(model_type, None)
    if model_cls is None:
        raise ValueError(f"Unknown model type {model_type}")
    model = model_cls(
        img_dim,
        latent_dim=latent_dim,
        in_channels=in_channels,
        num_feature=num_feature,
    )
    return model


MODELS = {
    "BernoulliMLPVAE": BernoulliMLPVAE,
    "BernoulliDCVAE": BernoulliDCVAE,
    "MixLogisticsDCVAE": MixLogisticsDCVAE,
    "DiagonalGaussianDCVAE": DiagonalGaussianDCVAE,
    "ConstantGaussianDCVAE": ConstantGaussianDCVAE,
    "SimpleRealNVP": SimpleRealNVP,
}
