import torch
import numpy as np
import random
torch.cuda.empty_cache()



import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from functools import partial
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../Models'))
from models.GANs import SimpleGANGenerator
from models.noise import get_noise_function
from models.utils import EHRDataset
from utils.helpers import MongoExtractor
from models.utils import load_checkpoint

import yaml
import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from functools import partial


from utils.helpers import MongoExtractor
from models.GANs import SimpleGANGenerator
from models.utils import EHRDataset
from train.train_DNN_v1_1 import collate_fn, preprocess_data

from dotenv import load_dotenv

load_dotenv()
config_file = '/home/andrea/Desktop/Sarcoma-DT/Models/configs/config_v1_1.yaml'
model_checkpoint = '/home/andrea/Desktop/Sarcoma-DT/Models/runs/run_2024-11-10_19-04-10/checkpoints/generator_epoch_99.pth'
validation_data_config = '/home/andrea/Desktop/Sarcoma-DT/Models/configs/config_v1_1_val.yaml'

def inference(generator, data_loader, device):
    generator.eval()
    print(f"Sum of generator parameters after loading: {sum(p.sum().item() for p in generator.parameters())}")

    results = []
    real_features_list = []
    labels_list = []
    with torch.no_grad():
        print(f"Sum of generator parameters after loading: {sum(p.sum().item() for p in generator.parameters())}")

        for features, labels in data_loader:
            real_features = features
            print(f"Sum of generator parameters after loading: {sum(p.sum().item() for p in generator.parameters())}")

            # Generate noise for inference
            # noise = noise_function(real_features.size(0), device)
       
            # Concatenate features and noise
            generator_input = real_features

            # Generate fake labels using the generator
            fake_labels = generator(generator_input)

            # Collect results
            results.append(fake_labels.cpu().numpy())
            real_features_list.append(real_features.cpu().numpy())
            labels_list.append(labels-1)

    return np.argmax(np.vstack(results), axis=1), np.vstack(labels_list).flatten().astype(int), real_features_list
config = MongoExtractor.load_config(config_file)

# Extract data configuration
dc = config['data_config']

# Load training data
mongo_extractor_train = MongoExtractor(
    connection_string=os.getenv('MONGO_URI'),
    database_name=os.getenv('MONGO_DB'),
    collection_name=os.getenv('MONGO_COLLECTION'),
    config_file=config_file
)
df = mongo_extractor_train.get_dataframe()

# Load validation data
mongo_extractor_val = MongoExtractor(
    connection_string=os.getenv('MONGO_URI'),
    database_name=os.getenv('MONGO_DB'),
    collection_name=os.getenv('MONGO_COLLECTION'),
    config_file=validation_data_config
)
df_val= mongo_extractor_val.get_dataframe()
df_val_original = mongo_extractor_val.get_dataframe()

dc = config['data_config']
mc = config['model_config']
sgc = config['simplegan_config']
mc['latent_dim'] = 0
mc['noise_type'] = 'uniform'
mc['batch_size'] = 62

df, df_val, scaler, label_encoder = preprocess_data(df, df_val, dc)

dataset_train = EHRDataset(df)
dataset_val = EHRDataset(df_val)

collate_fn_dc = partial(collate_fn, dc=dc)
data_loader_val = DataLoader(dataset_val, batch_size=mc['batch_size'], drop_last=True, collate_fn=collate_fn_dc)


exclude = list(set(dc['exclude_fields']) - set(dc['labels']))
df = df.drop(columns=exclude)
df = df.drop(columns=dc['labels'])
sgc['input_dim'] = df.shape[1] 

# Load the generator model and optimizer

generator = SimpleGANGenerator(input_dim=sgc['input_dim'] + mc['latent_dim'], output_dim=sgc['output_dim'])

device = torch.device('cpu')
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0001)  # Use the same optimizer as during training
generator, optimizer_gen, _, _ = load_checkpoint(model_checkpoint, generator, optimizer_gen)
checkpoint = torch.load(model_checkpoint)

generator.load_state_dict(checkpoint['model_state_dict'], strict=True)
optimizer_gen.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
generator = generator.to(device)



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# generator = generator.float()


categorical_fields = list(set(dc['categorical_fields']) - set(dc['exclude_fields']))
all_features = df.columns.tolist()
numerical_fields = list(set(all_features) - set(categorical_fields) - set(dc['exclude_fields']))
pred_labels, real_labels, features = inference(generator, data_loader_val, device)



# Convert Tensors to DataFrame

# Convert tensors to a list of lists
features= np.array(features)
features_reshaped = features.reshape(-1, features.shape[-1])

column_names = ['size_a', 'size_b', 'size_c', 'age_admission', 'male', 'female', 'group_B', 'group_D', 'group_S', 'metastasis_no', 'metastasis_si', 'intermediat', 'malign']

# Create DataFrame
df_validation = pd.DataFrame(features_reshaped, columns=column_names)


# Display the DataFrame
df_validation[['size_a', 'size_b', 'size_c', 'age_admission' ]] = scaler.inverse_transform(df_validation[[ 'size_a', 'size_b', 'size_c', 'age_admission' ]])
df_validation['real_labels'] = real_labels
df_validation['predicted_labels'] = pred_labels

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt



# Compute confusion matrix
conf_matrix = confusion_matrix(df_validation['real_labels'], df_validation['predicted_labels'])
accuracy = accuracy_score(df_validation['real_labels'], df_validation['predicted_labels'])
print(f'Accuracy: {accuracy:.2f}')
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix\n0: No Evidence of Disease, 1: Disease')
plt.show()