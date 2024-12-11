from pathlib import Path
import torch
from einops import reduce
import numpy as np
from typing import Tuple

from autoencoder import TopkSparseAutoencoder
from datalib import MySafeTensorDataset

def calculate_feature_density(
    sae: TopkSparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate feature density for each feature in the sparse autoencoder.
    
    Feature density is defined as:
    $$\text{density}_i = \frac{\sum_{j} \mathbb{1}[f_i(x_j) > 0]}{N}$$
    where:
    - $f_i(x_j)$ is the activation of feature i on input j
    - N is total number of tokens/inputs
    
    Args:
        sae: Trained sparse autoencoder model
        dataloader: DataLoader containing input activations
        device: Device to run computation on
        
    Returns:
        feature_densities: Tensor of shape [num_features] containing density per feature
        feature_activations: Mean activation values when features are active
    """
    sae.eval()
    sae.to(device)
    
    total_samples = 0
    activation_counts = torch.zeros(sae.pages, device=device)
    activation_sums = torch.zeros(sae.pages, device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]  # Extract inputs if batch contains labels
            else:
                x = batch
            
            x = x.to(device)
            batch_size = x.shape[0]
            
            # Get encoded activations
            encoded_acts = sae.encode(x)  # Shape: [batch_size, num_features]
            
            # Count non-zero activations
            activation_counts += (encoded_acts > 0).float().sum(dim=0)
            
            # Sum activation values when active
            activation_sums += torch.where(
                encoded_acts > 0,
                encoded_acts,
                torch.zeros_like(encoded_acts)
            ).sum(dim=0)
            
            total_samples += batch_size
            
    # Calculate densities
    feature_densities = activation_counts / total_samples
    
    # Calculate mean activation when feature is active
    feature_activations = torch.where(
        activation_counts > 0,
        activation_sums / activation_counts,
        torch.zeros_like(activation_sums)
    )
    
    return feature_densities, feature_activations

def analyze_density_distribution(feature_densities: torch.Tensor):
    """
    Analyze the distribution of feature densities to identify clusters.
    
    Args:
        feature_densities: Tensor of feature densities
    """
    # Convert to numpy for analysis
    densities = feature_densities.cpu().numpy()
    
    # Calculate log densities
    log_densities = np.log10(densities + 1e-10)  # Add small epsilon to handle zeros
    
    print(f"Density Statistics:")
    print(f"Total features: {len(densities)}")
    print(f"Dead features (density=0): {(densities == 0).sum()}")
    print(f"Mean log density: {log_densities.mean():.3f}")
    print(f"Std log density: {log_densities.std():.3f}")
    
    # Identify clusters based on paper's observations
    ultralow = (densities > 0) & (densities < 1e-6)
    high = densities >= 1e-5
    
    print(f"\nDensity Clusters:")
    print(f"Ultralow density features (<1e-6): {ultralow.sum()}")
    print(f"High density features (≥1e-5): {high.sum()}")

# Example usage:
if __name__ == "__main__":
    sae_dir = './checkpoints/gigapath_sae_wsi2k'
    features_file = './scripts/Gigapath_embeddings.safetensors'
    pid_file = './scripts/Gigapath_ids.txt'
 
    # Load model and data
    sae = TopkSparseAutoencoder.from_pretrained(sae_dir)
    with open(pid_file, 'r') as f:
        patient_ids = [line.strip() for line in f]
        patient_ids = np.array(patient_ids)
    dataset = MySafeTensorDataset(Path(features_file))
    print('Total number of samples:', len(dataset))

    # random sampling
    num_samples = 100000
    if num_samples > len(patient_ids):
        num_samples = len(patient_ids)
    
    # Generate random indices and Select random subset of patient_ids
    random_indices = np.random.choice(len(patient_ids), size=num_samples, replace=False)
    patient_ids = patient_ids[random_indices]
    dataset = torch.utils.data.Subset(dataset, random_indices)
    print(f'Randomly selected {num_samples} samples for visualization')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=0,
        shuffle=False
    )

    densities, mean_activations = calculate_feature_density(sae, dataloader)
    analyze_density_distribution(densities)