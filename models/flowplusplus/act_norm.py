import torch
import torch.nn as nn

from util import mean_dim


class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width):
        super(_BaseNorm, self).__init__()

        # Input gets concatenated along channel axis
        num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            mean, inv_std = self._get_moments(x)
            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, ldj=None, reverse=False, condition_embd=None):
        x = torch.cat(x, dim=1)
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        x = x.chunk(2, dim=1)

        return x, ldj


class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels):
        super(ActNorm, self).__init__(num_channels, 1, 1)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum() * x.size(2) * x.size(3)
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum() * x.size(2) * x.size(3)

        return x, sldj


class CondNorm(ActNorm):
    """Activation Normalization used in Glow

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels, condition_embd_size=0):
        super(CondNorm, self).__init__(num_channels)

        if condition_embd_size > 0:
            hidden_size = 50
            self.conditioning_hidden_projection = nn.Linear(condition_embd_size, hidden_size, bias=True)
            self.conditioning_output_projection = nn.Linear(hidden_size, 4 * num_channels, bias=True)

    def _center(self, x, mean, reverse=False):
        if reverse:
            return x + mean
        else:
            return x - mean

    def _scale(self, x, sldj, inv_std, reverse=False):
        if reverse:
            x = x / inv_std
            sldj = sldj - inv_std.log().sum(dim=1).mean() * x.size(2) * x.size(3)
        else:
            x = x * inv_std
            sldj = sldj + inv_std.log().sum(dim=1).mean() * x.size(2) * x.size(3)

        return x, sldj

    def forward(self, x, ldj=None, reverse=False, condition_embd=None):
        x = torch.cat(x, dim=1)

        if not self.is_initialized:
            self.initialize_parameters(x)

        if condition_embd is not None:
            embedding_hidden = torch.nn.functional.relu(self.conditioning_hidden_projection(condition_embd))
            embedding_projected = self.conditioning_output_projection(embedding_hidden)
            embedding_projected = embedding_projected.view(x.size(0), -1, 1, 1)
            conditioned_out = torch.tanh(embedding_projected)
            mean_d, inv_std_d = torch.chunk(conditioned_out, 2, dim=1)

            mean = self.mean + mean_d
            inv_std = self.inv_std + inv_std_d
        else:
            mean = self.mean
            inv_std = self.inv_std

        if reverse:
            x, ldj = self._scale(x, ldj, inv_std, reverse)
            x = self._center(x, mean, reverse)
        else:
            x = self._center(x, mean, reverse)
            x, ldj = self._scale(x, ldj, inv_std, reverse)
        x = x.chunk(2, dim=1)

        return x, ldj


class PixNorm(_BaseNorm):
    """Pixel-wise Activation Normalization used in Flow++

    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.
    """
    def _get_moments(self, x):
        mean = torch.mean(x.clone(), dim=0, keepdim=True)
        var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False, condition_embd=None):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum()
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum()

        return x, sldj
