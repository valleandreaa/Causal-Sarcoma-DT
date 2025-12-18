import numpy as np
import torch

def gaussian_noise(batch_size, latent_dim, mean=0.0, std=1.0):
    """
    Generates Gaussian noise.
    
    Parameters:
    - batch_size (int): Number of noise vectors to generate.
    - latent_dim (int): Dimensionality of the noise vector.
    - mean (float): Mean of the Gaussian distribution.
    - std (float): Standard deviation of the Gaussian distribution.
    
    Returns:
    - torch.Tensor: Gaussian noise tensor of shape (batch_size, latent_dim).
    """
    return torch.randn(batch_size, latent_dim) * std + mean

def uniform_noise(batch_size, latent_dim, low=-1.0, high=1.0):
    # torch.manual_seed(42)
    """
    Generates Uniform noise.
    
    Parameters:
    - batch_size (int): Number of noise vectors to generate.
    - latent_dim (int): Dimensionality of the noise vector.
    - low (float): Lower bound of the uniform distribution.
    - high (float): Upper bound of the uniform distribution.
    
    Returns:
    - torch.Tensor: Uniform noise tensor of shape (batch_size, latent_dim).
    """
    return torch.rand(batch_size, latent_dim) * (high - low) + low

def multivariate_gaussian_noise(batch_size, latent_dim, mean=None, cov=None):
    """
    Generates Multivariate Gaussian noise.
    
    Parameters:
    - batch_size (int): Number of noise vectors to generate.
    - latent_dim (int): Dimensionality of the noise vector.
    - mean (np.array): Mean vector of the Gaussian distribution.
    - cov (np.array): Covariance matrix of the Gaussian distribution.
    
    Returns:
    - torch.Tensor: Multivariate Gaussian noise tensor of shape (batch_size, latent_dim).
    """
    if mean is None:
        mean = np.zeros(latent_dim)
    if cov is None:
        cov = np.eye(latent_dim)
    noise = np.random.multivariate_normal(mean, cov, batch_size)
    return torch.tensor(noise, dtype=torch.float32)

def bernoulli_noise(batch_size, latent_dim, p=0.5):
    """
    Generates Bernoulli noise.
    
    Parameters:
    - batch_size (int): Number of noise vectors to generate.
    - latent_dim (int): Dimensionality of the noise vector.
    - p (float): Probability of sampling 1.
    
    Returns:
    - torch.Tensor: Bernoulli noise tensor of shape (batch_size, latent_dim).
    """
    return torch.bernoulli(torch.full((batch_size, latent_dim), p))

def custom_noise(batch_size, latent_dim, distribution_fn):
    """
    Generates custom noise using a provided distribution function.
    
    Parameters:
    - batch_size (int): Number of noise vectors to generate.
    - latent_dim (int): Dimensionality of the noise vector.
    - distribution_fn (callable): Function that generates noise.
    
    Returns:
    - torch.Tensor: Custom noise tensor of shape (batch_size, latent_dim).
    """
    noise = distribution_fn(batch_size, latent_dim)
    return torch.tensor(noise, dtype=torch.float32)

noise_functions = {
    "gaussian": gaussian_noise,
    "uniform": uniform_noise,
    "multivariate_gaussian": multivariate_gaussian_noise
}

def get_noise_function(mc):
    noise_fn = noise_functions.get(mc['noise_type'])
    
    if mc['noise_type'] == 'gaussian':
        noise_tensor = noise_fn(mc['batch_size'], mc['latent_dim'], 0.5, 0.1)
        
    elif mc['noise_type'] == 'uniform':
        noise_tensor = noise_fn(mc['batch_size'], mc['latent_dim'], mc.get('low', 0.0), mc.get('high', 1.0))

    elif mc['noise_type'] == 'multivariate_gaussian':
        mean_vector = mc.get('mean', np.zeros(mc['latent_dim']))
        cov_matrix = mc.get('cov', np.eye(mc['latent_dim']))
        noise_tensor = noise_fn(mc['batch_size'], mc['latent_dim'], mean_vector, cov_matrix)

    return noise_tensor
def generate_sequence_noise(batch_size, seq_len, feature_dim, noise_type="gaussian", device=None, **kwargs):
    """Generate noise for sequence models.

    Parameters
    ----------
    batch_size: int
        Number of samples in the batch.
    seq_len: int
        Sequence length.
    feature_dim: int
        Dimensionality of each feature vector in the sequence.
    noise_type: str, optional
        One of ``"gaussian"`` or ``"uniform"``. Defaults to ``"gaussian"``.
    device: torch.device, optional
        Device on which to create the noise tensor.
    **kwargs: dict
        Additional parameters passed to the underlying noise function.

    Returns
    -------
    torch.Tensor
        Noise tensor of shape ``(batch_size, seq_len, feature_dim)``.
    """
    device = device or torch.device("cpu")

    if noise_type == "gaussian":
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 1.0)
        noise = torch.randn(batch_size, seq_len, feature_dim, device=device) * std + mean
    elif noise_type == "uniform":
        low = kwargs.get("low", -1.0)
        high = kwargs.get("high", 1.0)
        noise = torch.rand(batch_size, seq_len, feature_dim, device=device) * (high - low) + low
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    return noise