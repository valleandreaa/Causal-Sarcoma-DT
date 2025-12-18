import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score


def inception_score(images, splits=10):
    """
    Computes the Inception Score (IS) for a set of images.
    
    Parameters:
    images (torch.Tensor): A batch of images.
    splits (int): Number of splits for calculating the score.
    
    Returns:
    float: The Inception Score.
    """
    N = len(images)
    assert N > splits

    # Load pre-trained InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    def get_pred(x):
        x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i in range(0, N, splits):
        batch = images[i:i + splits]
        preds[i:i + splits] = get_pred(batch)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def frechet_inception_distance(real_images, fake_images):
    """
    Computes the Frechet Inception Distance (FID) between real and fake images.
    
    Parameters:
    real_images (torch.Tensor): A batch of real images.
    fake_images (torch.Tensor): A batch of generated images.
    
    Returns:
    float: The FID score.
    """
    # Load pre-trained InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    def get_activations(images):
        images = up(images)
        with torch.no_grad():
            pred = inception_model(images)
        return pred.cpu().numpy()

    act1 = get_activations(real_images)
    act2 = get_activations(fake_images)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def mean_squared_error(real_images, fake_images):
    """
    Computes the Mean Squared Error (MSE) between real and fake images.
    
    Parameters:
    real_images (torch.Tensor): A batch of real images.
    fake_images (torch.Tensor): A batch of generated images.
    
    Returns:
    float: The MSE score.
    """
    return F.mse_loss(fake_images, real_images).item()

def peak_signal_to_noise_ratio(real_images, fake_images):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between real and fake images.
    
    Parameters:
    real_images (torch.Tensor): A batch of real images.
    fake_images (torch.Tensor): A batch of generated images.
    
    Returns:
    float: The PSNR score.
    """
    mse = mean_squared_error(real_images, fake_images)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def auc_roc(real_labels, predicted_scores):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
    
    Parameters:
    real_labels (numpy.ndarray): True binary labels.
    predicted_scores (numpy.ndarray): Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
    
    Returns:
    float: The AUC-ROC score.
    """
    return roc_auc_score(real_labels, predicted_scores)