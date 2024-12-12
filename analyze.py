from pathlib import Path
import pandas as pd
import torch
from einops import reduce
import numpy as np
import scipy.sparse as sp
from typing import Tuple

from autoencoder import TopkSparseAutoencoder
from datalib import MySafeTensorDataset

import torch
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from einops import reduce


def custom_repr(self):
    return f'{{Shape:{tuple(self.shape)}}} value:{original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

class FeatureAnalyzer:
    def __init__(self, sae: TopkSparseAutoencoder, device: str = "cuda"):
        """
        Initialize analyzer for finding samples that highly activate features.
        
        Args:
            sae: Trained sparse autoencoder model
            device: Device to run computation on
        """
        self.sae = sae.to(device)
        self.device = device
        self.sae.eval()

    def find_top_activating_samples(
        self, 
        dataloader: torch.utils.data.DataLoader,
        pids: np.array,
        num_top_samples: int = 20,
        feature_indices: List[int] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Find samples that most strongly activate each feature.
        
        Args:
            dataloader: DataLoader containing input samples
            pids: patient ids
            num_top_samples: Number of top activating samples to keep per feature
            feature_indices: Specific features to analyze. If None, analyze all.
            
        Returns:
            Dictionary mapping feature index to list of (pid, activation_value) pairs
        """
        if feature_indices is None:
            feature_indices = list(range(self.sae.pages))

        # Initialize storage for top samples
        top_samples = {
            idx: [] for idx in feature_indices
        }
        
        # Track minimum activation threshold for each feature
        min_activations = {
            idx: 0.0 for idx in feature_indices
        }

        total_iterations = len(dataloader) * len(feature_indices)
        progress_bar = tqdm(total=total_iterations, desc="Processing batches & features")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                feats, labels = batch
                x = feats.to(self.device)  # [batch_size, num_features]
                
                # Get feature activations
                encoded_feats = self.sae.encode(x)  # [batch_size, n_pages]
                # sparse_feats = sp.csr_matrix(encoded_feats.cpu().numpy())

                # For each feature we're tracking
                for feat_idx in feature_indices:
                    # feats = sparse_feats[:, feat_idx].toarray().flatten()  # [batch_size]
                    feats = encoded_feats[:, feat_idx]  # [batch_size]
                    
                    # Find samples above current minimum threshold
                    strong_activations = feats > min_activations[feat_idx] # [batch_size]
                    if not strong_activations.any():
                        progress_bar.update(1)
                        continue
                        
                    # Get the pids and their activation values
                    new_pids = pids[batch_idx * dataloader.batch_size : (batch_idx + 1) * dataloader.batch_size][strong_activations.cpu().numpy()]
                    new_values = feats[strong_activations] # [num_strong_samples] 
                    
                    # Add to current top samples - use efficient list extension
                    top_samples[feat_idx].extend(
                        (pid, value) for pid, value in zip(new_pids, new_values)
                    )
                    
                    # Sort and keep top num_top_samples
                    top_samples[feat_idx].sort(key=lambda x: x[1], reverse=True)
                    top_samples[feat_idx] = top_samples[feat_idx][:num_top_samples]
                    
                    # Update minimum threshold
                    min_activations[feat_idx] = top_samples[feat_idx][-1][1]

                    progress_bar.update(1)

            progress_bar.close()

        return top_samples

    def analyze_feature_patterns(
        self,
        top_samples_dict: Dict[int, List[Tuple[torch.Tensor, float]]],
        feature_indices: List[int] = None
    ) -> Dict[int, Dict]:
        """
        Analyze patterns in top activating samples for each feature.
        
        Args:
            top_samples_dict: Output from find_top_activating_samples
            feature_indices: Features to analyze. If None, analyze all.
            
        Returns:
            Dictionary containing analysis results for each feature
        """
        if feature_indices is None:
            feature_indices = list(top_samples_dict.keys())

        feat_stats = {}
        
        for feat_idx in tqdm(feature_indices, desc="Analyzing feature patterns"):
            if not top_samples_dict[feat_idx]:
                continue # skip if no top samples

            pids, values = zip(*top_samples_dict[feat_idx])
            values = torch.tensor(values)
            
            analysis = {
                "feat_idx": feat_idx,
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "pids": ','.join(map(str, pids)),
                "dist": np.round(values.numpy(), 4),
            }
            
            feat_stats[feat_idx] = analysis

        return feat_stats

    def save_to_csv(self, feat_stats):
        # save to a csv file
        df = pd.DataFrame(feat_stats).T
        df.to_csv('top_activations.csv', index=False)

def visualize_top_activations(
    feature_idx: int,
    top_samples_dict: Dict[int, List[Tuple[torch.Tensor, float]]],
    analysis_results: Dict[int, Dict]
):
    """
    Visualize the activation patterns for a specific feature.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot activation distribution
    values = analysis_results[feature_idx]["activation_distribution"]
    sns.histplot(values, ax=ax1)
    ax1.set_title(f"Feature {feature_idx} Activation Distribution")
    ax1.set_xlabel("Activation Value")
    
    # Plot top sample activations
    top_values = [v for _, v in top_samples_dict[feature_idx]]
    ax2.plot(range(len(top_values)), top_values, 'bo-')
    ax2.set_title(f"Top {len(top_values)} Activation Values")
    ax2.set_xlabel("Sample Rank")
    ax2.set_ylabel("Activation Value")
    
    plt.tight_layout()
    plt.show()

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
    num_samples = len(dataset)//5 # 4096*2 # 100000
    if num_samples > len(patient_ids):
        num_samples = len(patient_ids)
    
    # Generate random indices and Select random subset of patient_ids
    random_indices = np.random.choice(len(patient_ids), size=num_samples, replace=False)
    patient_ids = patient_ids[random_indices]
    dataset = torch.utils.data.Subset(dataset, random_indices)
    print(f'Randomly selected {num_samples} samples for visualization')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16384,
        num_workers=20,
        shuffle=False
    )

    # densities, mean_activations = calculate_feature_density(sae, dataloader)
    # analyze_density_distribution(densities)

    # Find top activating samples
    analyzer = FeatureAnalyzer(sae, 'cuda:1')
    top_samples = analyzer.find_top_activating_samples( dataloader, pids=patient_ids, num_top_samples=10)
    
    # Analyze patterns
    analysis_results = analyzer.analyze_feature_patterns(top_samples)
    analyzer.save_to_csv(analysis_results)
    
    # Visualize results for specific features
    # for feature_idx in [0, 1, 2]:  # Example features
    #     visualize_top_activations(feature_idx, top_samples, analysis_results)

