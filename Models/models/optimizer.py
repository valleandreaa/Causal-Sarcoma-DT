import torch.optim as optim
import torch.nn as nn


def get_optimizer_gen(config_model, model_parameters):
    optimizer_type = config_model.get('opt_gen', 'SGD')
    

    if optimizer_type == 'sgd':

        optimizer = optim.SGD(model_parameters, lr=config_model['learning_rate'])
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model_parameters, lr=config_model['learning_rate_gen'], weight_decay=1e-4) #DNN=1e-5


    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model_parameters,lr=config_model['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer

def get_optimizer_disc(config_model, model_parameters):

    optimizer_type = config_model.get('opt_disc', 'SGD')
    optimizer_params = config_model.get('params_opt_disc', {})

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model_parameters, **optimizer_params)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model_parameters, lr=config_model['learning_rate_disc'], weight_decay=1e-4)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model_parameters, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def get_criterion(config_model):

    criterion_type = config_model.get('criterion', 'CrossEntropyLoss')
    criterion_params = config_model.get('params_criterion', {})

    if criterion_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(**criterion_params)
    elif criterion_type == 'MSELoss':
        criterion = nn.MSELoss(**criterion_params)
    elif criterion_type == 'L1Loss':
        criterion = nn.L1Loss(**criterion_params)
    else:
        raise ValueError(f"Unsupported criterion type: {criterion_type}")

    return criterion