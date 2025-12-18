import yaml
import sys
import argparse
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils.helpers import MongoExtractor
from models.GANs import SimpleGANGenerator, SimpleGANDiscriminator
from models.noise import get_noise_function
from models.optimizer import get_optimizer_gen, get_optimizer_disc, get_criterion
from models.losses import adversarial_loss
from models.utils import EHRDataset
from functools import partial
import shutil

def collate_fn(batch, dc):
    # Convert the batch into a DataFrame
    df = pd.DataFrame(batch)

    # Drop the excluded fields from the DataFrame
    df = df.drop(columns=dc['exclude_fields'])

    # Separate the features and labels from the DataFrame
    features = df.drop(columns=dc['labels'])
    labels = df[dc['labels']]

    # Convert the features and labels into tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

    return features_tensor, labels_tensor

def train(generator, discriminator, features, labels, noise, optimizer_G, optimizer_D, adversarial_loss, device):
    # Set models to train mode
    generator.train()
    discriminator.train()

    # Move real features and noise to device
    real_features = features.to(device)
    noise = noise.to(device)

    # Convert labels to one-hot encoding
    labels = labels.clone().detach().to(dtype=torch.int64)
    labels = F.one_hot(labels, num_classes=10).float()
    labels = labels.squeeze(1)
    real_labels = labels.to(device)

    # Generate fake labels using generator
    generator_input = torch.cat((real_features, noise), dim=1)
    fake_labels = generator(generator_input)

    # Update discriminator
    optimizer_D.zero_grad()
    real_pair = torch.cat((real_features, real_labels), dim=1)
    real_output = discriminator(real_pair)
    real_loss = adversarial_loss(real_output, torch.ones(real_features.size(0), 1, device=device))
    fake_pair = torch.cat((real_features, fake_labels.detach()), dim=1)
    fake_output = discriminator(fake_pair)
    fake_loss = adversarial_loss(fake_output, torch.zeros(real_features.size(0), 1, device=device))
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # Update generator
    optimizer_G.zero_grad()
    fake_labels = generator(generator_input)
    fake_pair = torch.cat((real_features, fake_labels), dim=1)
    g_loss = adversarial_loss(discriminator(fake_pair), torch.ones(real_features.size(0), 1, device=device))
    g_loss.backward()
    optimizer_G.step()

    return g_loss.item(), d_loss.item()

def validate(generator, discriminator, data_loader, latent_dim, adversarial_loss, device):
    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()

    g_losses = []
    d_losses = []

    with torch.no_grad():
        for features, labels in data_loader:
            real_features = features.to(device)

            # Convert labels to one-hot encoding
            labels = labels.clone().detach().to(dtype=torch.int64)
            labels = F.one_hot(labels, num_classes=10).float()
            labels = labels.squeeze(1)
            real_labels = labels.to(device)

            # Generate noise
            noise = torch.randn(real_features.size(0), latent_dim, device=device)

            # Generate fake labels using generator
            generator_input = torch.cat((real_features, noise), dim=1)
            fake_labels = generator(generator_input)

            # Calculate real loss
            real_pair = torch.cat((real_features, real_labels), dim=1)
            real_output = discriminator(real_pair)
            real_loss = adversarial_loss(real_output, torch.ones(real_features.size(0), 1, device=device))

            # Calculate fake loss
            fake_pair = torch.cat((real_features, fake_labels), dim=1)
            fake_output = discriminator(fake_pair)
            fake_loss = adversarial_loss(fake_output, torch.zeros(real_features.size(0), 1, device=device))

            # Calculate discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            # Calculate generator loss
            fake_pair = torch.cat((real_features, fake_labels), dim=1)
            g_loss = adversarial_loss(discriminator(fake_pair), torch.ones(real_features.size(0), 1, device=device))

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

    avg_g_loss = sum(g_losses) / len(g_losses)
    avg_d_loss = sum(d_losses) / len(d_losses)

    print(f"Validation Loss - Generator: {avg_g_loss:.4f}, Discriminator: {avg_d_loss:.4f}")

    return avg_g_loss, avg_d_loss

def main(config_file, checkpoint_dir, log_dir):
    # Load configuration file
    config = MongoExtractor.load_config(config_file)

    # Initialize MongoExtractor
    mongo_extractor = MongoExtractor(
        connection_string='mongodb://localhost:27017/',
        database_name='SarcomaDB',
        collection_name='Patients',
        config_file=config_file
    )

    # Get dataframe from MongoExtractor
    df = mongo_extractor.get_dataframe()

    # Extract relevant configurations
    dc = config['data_config']
    
    mc = config['model_config']
    sgc = config['simplegan_config']

    # Initialize scaler and label encoder
    scaler = StandardScaler()
    label_encoder = OneHotEncoder(sparse_output=True)
    
    # Scale age_at_admission column
    categorical_fields = dc['categorical_fields']

    # Assuming df is your DataFrame
    all_features = df.columns.tolist()

    # Find the difference
    numerical_fields = list(set(all_features) - set(categorical_fields) )
    
    df[numerical_fields] = scaler.fit_transform(df[numerical_fields])

    # Encode categorical columns
    categorical_fields = list(set(dc['categorical_fields'] ) - set(dc['labels'])) 
    for column in dc['categorical_fields']:
        encoded = label_encoder.fit_transform(df[[column]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=label_encoder.get_feature_names_out([column]))
        df = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)


    # Create EHRDataset and DataLoader
    dataset = EHRDataset(df)
    collate_fn_dc = partial(collate_fn, dc=dc)
    data_loader = DataLoader(dataset, batch_size=mc['batch_size'], drop_last=True, collate_fn=collate_fn_dc)

    # Get noise function
    noise = get_noise_function(mc)

    # Initialize generator and discriminator
    sgc['input_dim'] = df.shape[1] - len(dc['labels'])
    gen = SimpleGANGenerator(input_dim=sgc['input_dim']+mc['latent_dim'], output_dim=sgc['output_dim'])
    dis = SimpleGANDiscriminator(input_dim=sgc['input_dim']+sgc['output_dim'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set criterion
    criterion = adversarial_loss

    # Get optimizer for generator and discriminator
    optimizer_gen = get_optimizer_gen(mc, gen.parameters())
    optimizer_dis = get_optimizer_disc(mc, dis.parameters())

    # Set number of epochs
    num_epochs = mc['num_epochs']

    # Initialize lists to store losses
    g_losses = []
    d_losses = []

    # Training loop
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(data_loader):
            # Train generator and discriminator
            g_loss, d_loss = train(gen, dis, features, labels, noise, optimizer_gen, optimizer_dis, criterion, device)
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(data_loader)}] Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}")

        # Validate generator and discriminator
        avg_g_loss, avg_d_loss = validate(gen, dis, data_loader, mc['latent_dim'], criterion, device)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Save generator and discriminator checkpoints
        torch.save(gen.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth'))
        torch.save(dis.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))

    # Plot and save training losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'training_losses.png'))
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a GAN model.')
    parser.add_argument('--config_file', type=str, default='configs/config_m1_iter1_simpleGAN.yaml', help='Path to the configuration file.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs.')
    args = parser.parse_args()

    # Create checkpoint and log directories if they don't exist
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir)

    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.makedirs(args.log_dir)

    # Run main function
    main(args.config_file, args.checkpoint_dir, args.log_dir)