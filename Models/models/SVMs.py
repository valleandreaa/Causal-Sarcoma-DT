import torch
import torch.nn as nn

# SVM Classifier Model
class SVMClassifier(nn.Module):
    def __init__(self, input_size):
        super(SVMClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Hinge Loss for Classification
def hinge_loss_classification(output, target):
    return torch.mean(torch.clamp(1 - output * target, min=0))

# SVM Regressor Model
class SVMRegressor(nn.Module):
    def __init__(self, input_size):
        super(SVMRegressor, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Epsilon-insensitive Loss for Regression
def epsilon_insensitive_loss(output, target, epsilon=0.1):
    return torch.mean(torch.clamp(torch.abs(output - target) - epsilon, min=0))
