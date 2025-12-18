import torch
import torch.nn as nn
import json 
from models.encoder import Encoder 
from pathlib import Path
from datetime import datetime

class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)

class InstanceNorm(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features)

    def forward(self, x):
        return self.inorm(x)

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm, self).__init__()
        self.lnorm = nn.LayerNorm(num_features)

    def forward(self, x):
        return self.lnorm(x)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.matmul(w.view(height, -1).data.t(), u.data))
            u.data = self._l2normalize(torch.matmul(w.view(height, -1).data, v.data))

        sigma = torch.dot(u.data, torch.matmul(w.view(height, -1).data, v.data))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    

def randomize_one_hot(one_hot: torch.Tensor) -> torch.Tensor:
    """
    Given a one‐hot tensor of shape [..., C], returns a new one‐hot tensor
    where for each “row” (i.e. each index in the flattened leading dims),
    the '1' has been moved to a *different* random channel.

    Args:
      one_hot: Tensor of shape [d1, d2, ..., dk, C], with exactly one '1' per
               final-dimension slice, and zeros elsewhere.

    Returns:
      new_one_hot: Tensor of same shape/device, with each slice’s '1' moved
                   to a different random index in [0..C-1].
    """
    # Remember original shape
    orig_shape = one_hot.shape              # e.g. (16, 898, 3)
    C = orig_shape[-1]                      # number of channels (here 3)

    # Flatten all but the last dim: result is [N, C] where N = product(orig_shape[:-1])
    flat = one_hot.view(-1, C)              # e.g. [16*898, 3]
    N = flat.size(0)

    # 1) find original '1' positions in each row
    orig_idx = flat.argmax(dim=1)           # [N], values in 0..C-1

    # 2) randomly pick a new index in [0..C-2] for each row
    r = torch.randint(0, C-1, (N,), device=one_hot.device)
    # 3) shift up by 1 whenever r >= orig_idx to skip the original channel
    new_idx = r + (r >= orig_idx).long()    # [N], now in 0..C-1 but != orig_idx

    # 4) build new one‐hot: zeros, then set the new positions to 1
    new_flat = torch.zeros_like(flat)       # [N, C]
    new_flat[torch.arange(N, device=one_hot.device), new_idx] = 1

    # 5) reshape back to original [..., C]
    return new_flat.view(orig_shape)

def randomize_multiple_one_hot(one_hot: torch.Tensor, extra_ones: int = 0) -> torch.Tensor:
    """
    Given a tensor of shape [..., C] with multiple '1's possible per row,
    returns a new tensor where for each "row" (i.e. each index in the 
    flattened leading dims), each '1' has been moved to a *different* 
    random channel.

    Args:
      one_hot: Tensor of shape [d1, d2, ..., dk, C], with one or more '1's per
               final-dimension slice, and zeros elsewhere.

    Returns:
      new_tensor: Tensor of same shape/device, with each slice's '1's moved
                  to different random indices in [0..C-1].
    """
    # Remember original shape
    orig_shape = one_hot.shape              # e.g. (16, 898, 3)
    C = orig_shape[-1]                      # number of channels (here 3)

    # Flatten all but the last dim: result is [N, C] where N = product(orig_shape[:-1])
    flat = one_hot.view(-1, C)              # e.g. [16*898, 3]
    N = flat.size(0)

    # Initialize new tensor with zeros
    new_flat = torch.zeros_like(flat)

    for i in range(N):
        # Find all positions with '1's in this row (and optionally add extra ones)
        orig_positions = torch.where(flat[i] == 1)[0]
        num_ones = len(orig_positions)
        
        # If no original ones and no extras requested, skip
        if num_ones == 0 and extra_ones == 0:
            continue
        # Determine how many ones to place (original + extra)
        total_ones = min(num_ones + extra_ones, C)
        # Randomly sample distinct new positions
        perm = torch.randperm(C, device=one_hot.device)
        new_positions = perm[:total_ones]
        
        # Set the new positions to 1
        new_flat[i, new_positions] = 1

    # Reshape back to original [..., C]
    return new_flat.view(orig_shape)

    
def randomize_binary_vector(binary_vector: torch.Tensor) -> torch.Tensor:
    """
    Given a binary tensor of shape [..., C], returns a new binary tensor
    where each row is randomly set to one of [1,0,0], [1,0,1], [1,1,0], or [1,1,1].
    For temporal data with shape [B, T, C], ensures all timesteps for each patient 
    have the same treatment pattern.

    Args:
      binary_vector: Tensor of shape [d1, d2, ..., dk, C], with binary values (0s and 1s).

    Returns:
      new_binary: Tensor of same shape/device, with each slice having a
                  randomly chosen pattern from the predefined options.
                  For 3D tensors (B, T, C), all timesteps T for each batch B 
                  will have the same pattern.
    """
    # Remember original shape
    orig_shape = binary_vector.shape
    C = orig_shape[-1]
    
    # Define the possible patterns
    patterns = torch.tensor([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ], device=binary_vector.device, dtype=binary_vector.dtype)
    
    # Handle 3D tensors (B, T, C) - ensure consistency across timesteps for each patient
    if len(orig_shape) == 3:
        B, T, C = orig_shape
        
        # Randomly choose one pattern per patient (batch element)
        pattern_indices = torch.randint(0, len(patterns), (B,))
        chosen_patterns = patterns[pattern_indices]  # Shape: [B, 3]
        
        # Ensure the chosen patterns match the expected size C
        if C != 3:
            if C > 3:
                # Pad with zeros
                padding = torch.zeros(B, C - 3, device=binary_vector.device, dtype=binary_vector.dtype)
                chosen_patterns = torch.cat([chosen_patterns, padding], dim=1)
            else:
                # Truncate
                chosen_patterns = chosen_patterns[:, :C]
        
        # Expand to all timesteps: [B, 1, C] -> [B, T, C]
        chosen_patterns = chosen_patterns.unsqueeze(1).expand(B, T, C)
        return chosen_patterns
    
    else:
        # For other shapes, use original logic
        # Calculate total number of vectors 
        total_elements = torch.prod(torch.tensor(orig_shape[:-1]))
        
        # Randomly choose a pattern for each vector
        pattern_indices = torch.randint(0, len(patterns), (total_elements,))
        chosen_patterns = patterns[pattern_indices]  # Shape: [total_elements, 3]
        
        # Ensure the chosen patterns match the expected size C
        if C != 3:
            if C > 3:
                # Pad with zeros
                padding = torch.zeros(total_elements, C - 3, device=binary_vector.device, dtype=binary_vector.dtype)
                chosen_patterns = torch.cat([chosen_patterns, padding], dim=1)
            else:
                # Truncate
                chosen_patterns = chosen_patterns[:, :C]
        
        # Reshape back to original shape
        return chosen_patterns.view(orig_shape)

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(file_path, model, optimizer):
    checkpoint = torch.load(file_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {file_path}")
    return model, optimizer, epoch, loss

def save_encoder_decoder(
        encoder,
        decoder,
        directory,
        metadata=None,
        feature_spec=None,
        one_hot_mappings=None,
        ord_mappings=None,
        model_params=None,
        numeric_ranges=None,
    ):
    """
    Saves encoder and decoder models along with metadata, feature specification,
    one_hot mappings, and model parameters.
    
    Parameters:
    - encoder: The encoder model (torch.nn.Module)
    - decoder: The decoder model (torch.nn.Module)
    - directory: Path to save the models
    - metadata: Dictionary with additional model info
    - feature_spec: Dictionary containing feature specification
    - one_hot_mappings: Dictionary mapping one_hot feature names to list of categories
    - model_params: Dictionary containing model parameters (e.g., hidden_dim, epoch, etc.)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    encoder_path = os.path.join(directory, f"encoder_{timestamp}.pt")
    decoder_path = os.path.join(directory, f"decoder_{timestamp}.pt")
    metadata_path = os.path.join(directory, f"metadata_{timestamp}.json")

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    model_metadata = {
        "encoder_path": encoder_path,
        "decoder_path": decoder_path,
        "encoder_class": encoder.__class__.__name__,
        "decoder_class": decoder.__class__.__name__,
        "timestamp": timestamp,
    }

    if feature_spec is not None:
        model_metadata["feature_spec"] = feature_spec
    if one_hot_mappings is not None:
        model_metadata["one_hot_mappings"] = one_hot_mappings
    if ord_mappings is not None:
        model_metadata["ord_mappings"] = ord_mappings
    if metadata:
        model_metadata.update(metadata)
    if model_params:
        model_metadata["model_params"] = model_params
    if numeric_ranges is not None:
        model_metadata["numeric_ranges"] = numeric_ranges

    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=4)

    print(f"Encoder, Decoder, and metadata saved in {directory}")

def save_temporal_cycle_gan(gx, gy, dx, dy, directory, metadata=None, model_params=None, timestamp=None):
    """Save CycleGAN components and metadata for later inference."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


    gx_path = directory / f"Gx.pt"
    gy_path = directory / f"Gy.pt"
    dx_path = directory / f"Dx.pt"
    dy_path = directory / f"Dy.pt"
    meta_path = directory / f"metadata.json"

    torch.save(gx.state_dict(), gx_path)
    torch.save(gy.state_dict(), gy_path)
    torch.save(dx.state_dict(), dx_path)
    torch.save(dy.state_dict(), dy_path)

    meta = {
        "Gx_path": str(gx_path),
        "Gy_path": str(gy_path),
        "Dx_path": str(dx_path),
        "Dy_path": str(dy_path),
        "Gx_class": gx.__class__.__name__,
        "Gy_class": gy.__class__.__name__,
        "Dx_class": dx.__class__.__name__,
        "Dy_class": dy.__class__.__name__,
        "timestamp": timestamp,
    }

    if metadata:
        meta.update(metadata)
    if model_params:
        meta["model_params"] = model_params

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print(f"Temporal CycleGAN models and metadata saved to {directory}")

class EHRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Returns a single row of the dataframe as a dict
        row = self.dataframe.iloc[idx]
        return row

from datetime import datetime
import os 

def create_run_folder(base_dir=None):
    """Create a run folder based on a timestamp or use the provided one."""
    if base_dir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_dir = os.path.join('./Models/runs', f'run_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

class MetadataHandler:
    """Class to handle metadata operations."""
    def __init__(self, metadata_clinical_file):
        """Initialize the MetadataHandler with a clinical metadata file."""
        self.metadata_clinical_file = metadata_clinical_file
        self.metadata = self.import_metadata()
        self.input_dim = self.metadata['model_params'].get("input_dim")
        self.hidden_dim = self.metadata['model_params'].get("hidden_dim")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ord_cards = self.metadata['model_params'].get("ord_cards")
        self.numeric_ranges = self.metadata.get("numeric_ranges", {})
        self.ord_mappings = self.metadata.get("ord_mappings", {})

        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.ord_cards).to(self.device)

        # Load pretrained weights if a path is provided in metadata
        enc_path = self.metadata.get("encoder_path")
        if enc_path and os.path.isfile(enc_path):
            self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
            self.encoder.eval()
        else:
            if enc_path:
                print(f"Warning: encoder weights not found at {enc_path}; using untrained encoder")

    def import_metadata(self):
        """Import metadata from a JSON file."""
        with open(self.metadata_clinical_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def save_metadata(self, metadata):
        """Save metadata to the clinical metadata file."""
        with open(self.metadata_clinical_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def get_feature_spec(self, cfg=None):
        """
        Retrieve the feature specification from metadata or configuration.

        Parameters:
        - cfg: Configuration object containing data_config

        Returns:
        - feature_spec: The feature specification dictionary
        """
        meta_clin = self.import_metadata()
        feature_spec = meta_clin.get("feature_spec")
        if feature_spec is None:
            feature_spec = cfg.data_config["features"]
        return feature_spec

    def get_encoded_dataset(self, dataset, device):
        """
        Instantiate an Encoder using dataset.input_dim (or tensor shape) and parameters from metadata,
        and move it to the specified device.
        Expects 'hidden_dim' under metadata['model_params'].
        """
        metadata = self.import_metadata()
        feature_spec = metadata.get("feature_spec", {})
        num_cols = [c for c, t in feature_spec.items() if t == "numeric"]

        x_num_oh = dataset[0].clone().to(device)
        x_ord = dataset[1].to(device) if dataset[1] is not None else None
        mask = dataset[2].to(device) if len(dataset) > 2 and dataset[2] is not None else None

        if mask is not None:
            x_num_oh = x_num_oh * mask[:, : x_num_oh.size(1)]
        if num_cols and self.numeric_ranges:
            for i, col in enumerate(num_cols):
                if col in self.numeric_ranges:
                    mn, mx = self.numeric_ranges[col]
                    if mx > mn:
                        x_num_oh[:, i] = (x_num_oh[:, i] - mn) / (mx - mn)

        encoded_dataset = self.encoder(x_num_oh, x_ord)
        return (encoded_dataset, mask) if mask is not None else encoded_dataset


