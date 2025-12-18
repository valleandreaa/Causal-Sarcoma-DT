class FeatureEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_sub_features, main_feature_dims):
        super(FeatureEmbedding, self).__init__()
        
        # Shared layer for sub-features
        self.shared_layer = nn.Linear(main_feature_dims['sub_feature_dim'], hidden_dim)
        
        # Separate layers for individual features
        self.layers_single_features = nn.ModuleDict({
            feature_name: nn.Linear(dim, hidden_dim) 
            for feature_name, dim in main_feature_dims['single_features'].items()
        })
        
        self.num_sub_features = num_sub_features
    
    def forward(self, x):
        # Process sub-features
        sub_feature_size = main_feature_dims['sub_feature_dim']
        sub_feature_outputs = []
        for i in range(self.num_sub_features):
            sub_feature = x[:, i*sub_feature_size:(i+1)*sub_feature_size]
            shared_output = F.relu(self.shared_layer(sub_feature))
            sub_feature_outputs.append(shared_output)
        
        # Process single features
        single_feature_outputs = []
        start_idx = self.num_sub_features * sub_feature_size
        for feature_name, layer in self.layers_single_features.items():
            end_idx = start_idx + main_feature_dims['single_features'][feature_name]
            single_feature = x[:, start_idx:end_idx]
            single_feature_output = F.relu(layer(single_feature))
            single_feature_outputs.append(single_feature_output)
            start_idx = end_idx
        
        # Concatenate all feature outputs
        concatenated_features = torch.cat(sub_feature_outputs + single_feature_outputs, dim=1)
        
        return concatenated_features
