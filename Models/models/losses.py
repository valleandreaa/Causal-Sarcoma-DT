import torch
import torch.nn as nn

def wasserstein_loss(real_output, fake_output):
    """
    Computes the Wasserstein loss for WGAN.
    
    Parameters:
    - real_output (torch.Tensor): Discriminator output for real images.
    - fake_output (torch.Tensor): Discriminator output for fake images.
    
    Returns:
    - torch.Tensor: Wasserstein loss.
    """
    return torch.mean(fake_output) - torch.mean(real_output)

def cycle_consistency_loss(real_image, reconstructed_image, lambda_cycle=10.0):
    """
    Computes the cycle consistency loss for CycleGAN.
    
    Parameters:
    - real_image (torch.Tensor): Original image.
    - reconstructed_image (torch.Tensor): Reconstructed image after cycle.
    - lambda_cycle (float): Weight for the cycle consistency loss.
    
    Returns:
    - torch.Tensor: Cycle consistency loss.
    """
    return lambda_cycle * nn.L1Loss()(real_image, reconstructed_image)

def identity_loss(real_image, same_image, lambda_identity=5.0):
    """
    Computes the identity loss for CycleGAN.
    
    Parameters:
    - real_image (torch.Tensor): Original image.
    - same_image (torch.Tensor): Image after passing through the generator.
    - lambda_identity (float): Weight for the identity loss.
    
    Returns:
    - torch.Tensor: Identity loss.
    """
    return lambda_identity * nn.L1Loss()(real_image, same_image)

def adversarial_loss(discriminator_output, target):
    """
    Computes the adversarial loss for GANs.
    
    Parameters:
    - discriminator_output (torch.Tensor): Discriminator output.
    - target (torch.Tensor): Target labels (1 for real, 0 for fake).
    
    Returns:
    - torch.Tensor: Adversarial loss.
    """
    target = target.to(discriminator_output.device)
    return nn.BCEWithLogitsLoss()(discriminator_output, target)
   

def least_squares_gan_loss(discriminator_output, target):
    """
    Computes the least squares loss for LSGAN.
    
    Parameters:
    - discriminator_output (torch.Tensor): Discriminator output.
    - target (torch.Tensor): Target labels (1 for real, 0 for fake).
    
    Returns:
    - torch.Tensor: Least squares loss.
    """
    return nn.MSELoss()(discriminator_output, target)

def hinge_loss(discriminator_output, target_is_real):
    """
    Computes the hinge loss for GANs.
    
    Parameters:
    - discriminator_output (torch.Tensor): Discriminator output.
    - target_is_real (bool): Whether the target is real or fake.
    
    Returns:
    - torch.Tensor: Hinge loss.
    """
    if target_is_real:
        return torch.mean(torch.relu(1.0 - discriminator_output))
    else:
        return torch.mean(torch.relu(1.0 + discriminator_output))

def feature_matching_loss(real_features, fake_features):
    """
    Computes the feature matching loss for GANs.
    
    Parameters:
    - real_features (torch.Tensor): Features from real images.
    - fake_features (torch.Tensor): Features from fake images.
    
    Returns:
    - torch.Tensor: Feature matching loss.
    """
    return nn.L1Loss()(real_features, fake_features)

# Mapping of adversarial loss functions
adversarial_loss_map = {
    "bce": adversarial_loss,
    "lsgan": least_squares_gan_loss,
        # ``target`` is typically a tensor of ones or zeros.  Convert to a boolean
    # by averaging across elements and checking if it represents real samples.
    "hinge": lambda output, target: hinge_loss(
        output, target.float().mean().item() > 0.5
    ),
    "wasserstein": wasserstein_loss,
}


def get_adversarial_loss(name: str):
    """Retrieve an adversarial loss function by name."""
    key = name.lower()
    if key not in adversarial_loss_map:
        raise ValueError(f"Unknown adversarial loss: {name}")
    return adversarial_loss_map[key]
