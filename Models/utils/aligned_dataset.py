
def create_aligned_clinical_dataset(config_file: str, clinical_config_key: str = "clinical_data_config"):
    """
    Create a clinical dataset using the EXACT same processing as train_encoder.py.
    This ensures perfect alignment between training and inference.
    
    Args:
        config_file: Path to configuration file
        clinical_config_key: Key in config for clinical data configuration
    
    Returns:
        TabularDataset instance with consistent processing
    """
    from encoder.train_encoder import TabularDataset
    from utils.helpers import MongoExtractor
    import yaml
    import os
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    clinical_config = config[clinical_config_key]
    
    # Create extractor
    extractor = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=None,
        custom_config=clinical_config
    )
    
    # Get dataframe
    df = extractor.get_dataframe()
    
    # CRITICAL: Reorder DataFrame columns to match feature_spec order
    feature_spec = clinical_config['features']
    df = MongoExtractor.reorder_dataframe_columns(df, feature_spec)
    
    # Create dataset using the SAME TabularDataset as training
    dataset = TabularDataset(df, feature_spec)
    
    return dataset, df

def load_encoder_with_correct_metadata(metadata_path: str, encoder_path: str, device: torch.device):
    """
    Load encoder with metadata validation to ensure dimension consistency.
    """
    import json
    from models.encoder import Encoder
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    params = metadata['model_params']
    input_dim = params['input_dim']
    hidden_dim = params['hidden_dim']
    
    # Load encoder
    encoder = Encoder(input_dim, hidden_dim).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    
    return encoder, metadata

def validate_feature_alignment(dataset, metadata):
    """
    Validate that dataset features match saved metadata.
    """
    saved_feature_spec = metadata.get('feature_spec', {})
    dataset_feature_spec = dataset.spec
    
    saved_keys = list(saved_feature_spec.keys())
    dataset_keys = list(dataset_feature_spec.keys())
    
    if saved_keys != dataset_keys:
        print("❌ FEATURE ALIGNMENT ERROR:")
        print(f"   Saved metadata features: {len(saved_keys)}")
        print(f"   Dataset features: {len(dataset_keys)}")
        
        # Show first difference
        for i, (saved, current) in enumerate(zip(saved_keys, dataset_keys)):
            if saved != current:
                print(f"   First difference at position {i}:")
                print(f"     Saved: '{saved}'")
                print(f"     Current: '{current}'")
                break
        
        return False
    else:
        print("✅ Feature alignment validated successfully")
        return True

def test_encoder_decoder_alignment():
    """
    Test encoder-decoder alignment with a simple reconstruction test.
    """
    import torch
    from models.encoder import Encoder
    from models.decoder import Decoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    metadata_path = "/home/andreavalle/Sarcoma-DT/saved_models/clinical_data_encoder_max/metadata.json"
    encoder_path = "/home/andreavalle/Sarcoma-DT/saved_models/clinical_data_encoder_max/encoder.pth"
    decoder_path = "/home/andreavalle/Sarcoma-DT/saved_models/clinical_data_encoder_max/decoder.pth"
    config_file = "/home/andreavalle/Sarcoma-DT/Models/configs/config_encoder_decoder_clinical_data_reordered.yaml"
    
    # Check if files exist
    if not all([Path(metadata_path).exists(), Path(encoder_path).exists(), Path(decoder_path).exists()]):
        print("❌ Missing saved model files - train encoder first")
        return False
    
    try:
        # Load encoder with validation
        encoder, metadata = load_encoder_with_correct_metadata(metadata_path, encoder_path, device)
        
        # Create aligned dataset
        dataset, df = create_aligned_clinical_dataset(config_file, "data_config")
        
        # Validate alignment
        if not validate_feature_alignment(dataset, metadata):
            return False
        
        # Load decoder
        params = metadata['model_params']
        hidden_dim = params['hidden_dim']
        output_dim = params.get('output_dim', params['input_dim'])
        
        decoder = Decoder(hidden_dim, output_dim).to(device)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.eval()
        
        # Test with first sample
        if len(dataset) > 0:
            x_target, x_ord, mask = dataset[0]
            x_target = x_target.unsqueeze(0).to(device)
            x_ord = x_ord.unsqueeze(0).to(device) if x_ord is not None else None
            mask = mask.unsqueeze(0).to(device)
            
            # Get encoder input (numeric + one-hot only)
            n_numeric = len(dataset.num_cols)
            n_onehot = sum(len(dataset.one_hot_mappings.get(col, [])) for col in dataset.oh_cols)
            x_input = x_target[:, :n_numeric + n_onehot]
            
            with torch.no_grad():
                # Encode
                embedding = encoder(x_input, x_ord)
                
                # Decode
                reconstruction = decoder(embedding)
                
                # Check dimensions
                print(f"✅ Reconstruction test passed:")
                print(f"   Input shape: {x_input.shape}")
                print(f"   Embedding shape: {embedding.shape}")
                print(f"   Reconstruction shape: {reconstruction.shape}")
                print(f"   Target shape: {x_target.shape}")
                
                if reconstruction.shape == x_target.shape:
                    print("✅ Perfect dimension alignment!")
                    return True
                else:
                    print("❌ Dimension mismatch in reconstruction")
                    return False
        
    except Exception as e:
        print(f"❌ Error in alignment test: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("TESTING ENCODER-DECODER ALIGNMENT")
    print("="*80)
    
    success = test_encoder_decoder_alignment()
    
    if success:
        print("\n✅ ALL TESTS PASSED - Alignment is correct!")
    else:
        print("\n❌ ALIGNMENT ISSUES DETECTED")
        print("   1. Ensure encoder is trained with config_encoder_decoder_clinical_data_reordered.yaml")
        print("   2. Use the create_aligned_clinical_dataset function in your code")
        print("   3. Always validate feature alignment before processing")
