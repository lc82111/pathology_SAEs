import matplotlib.pyplot as plt
import sklearn
import torch
import cuml.manifold
import scipy
import scipy.sparse
import wandb
import torch
import numpy as np
import umap
from tqdm import tqdm
from pathlib import Path
from autoencoder import TopkSparseAutoencoder
from datalib import SafeTensorDataset
import cuml
import cupy as cp

def create_sparse_array(tops_acts, top_indices, num_pages):
    """
    Create sparse cupy array from top-K activations
    
    Args:
        tops_acts: torch tensor of shape (B, K) - activation values
        top_indices: torch tensor of shape (B, K) - indices of activations
        num_pages: int - total number of pages/dictionary size (P)
    
    Returns:
        cp.sparse.csr_matrix of shape (B, P)
    """
    batch_size = tops_acts.shape[0]
    
    # Convert to numpy arrays
    values = tops_acts.cpu().numpy()
    col_indices = top_indices.cpu().numpy()
    
    # Create row indices
    row_indices = np.repeat(np.arange(batch_size), col_indices.shape[1])
    row_indices = np.asarray(row_indices)
    
    # Flatten the arrays
    values = np.asarray(values.flatten())
    col_indices = np.asarray(col_indices.flatten())
    
    # Create COO matrix
    sparse_array = scipy.sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(batch_size, num_pages)
    )
    
    # Convert to CSR format for better performance
    return sparse_array.tocsr()

def umap_reduce(
    sae: TopkSparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    n_components: int = 2,
    num_samples: int = 1000,
    neighbors: int = 30,  # Increased from 15
    min_dist: float = 0.0,    # Decreased from 0.1
    spread: float = 1.0,
    num_epochs: int = 500,  # Increased from 200
    save_file: str = 'umap_embeddings.npy'
):
    """
    Creates an improved UMAP visualization of SAE features.
    
    Args:
        sae: Trained TopK Sparse Autoencoder model
        dataloader: DataLoader for the dataset
        num_samples: Number of samples to visualize
        neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance between points in UMAP
        spread: Spread of the points in UMAP
    """
    sparse_features = []
    activation_data = {
        'acts': [],
        'indices': []
    }
    
    sae.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae.to(device)
    
    # Extract features
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAE features"):
            batch = batch.to(device)
            encoded_acts, tops_acts, top_indices = sae.encode(batch, return_topk=True)

            encoded_acts_sparse = create_sparse_array(
                tops_acts=tops_acts,
                top_indices=top_indices,
                num_pages=sae.pages
            )
            sparse_features.append(encoded_acts_sparse)

            activation_data['acts'].append(tops_acts.cpu())
            activation_data['indices'].append(top_indices.cpu())
            
            if sum(sf.shape[0] for sf in sparse_features) >= num_samples:
                break

    # Free memory
    sae.to('cpu')
    torch.cuda.empty_cache()

    # Combine sparse features
    features_sparse = scipy.sparse.vstack(sparse_features)
    features_sparse = features_sparse[:num_samples]

    # Normalize features
    normalizer = sklearn.preprocessing.Normalizer(norm='l2')
    features_normalized = normalizer.transform(features_sparse)

    # Improved UMAP parameters
    umap_reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        metric='cosine',
        n_epochs=num_epochs,  # Increased number of epochs
        learning_rate=1.0,
        init='spectral'  # Using spectral initialization
    )
    
    embeddings = umap_reducer.fit_transform(features_normalized)
    print("UMAP embeddings shape:", embeddings.shape)

    np.save(save_file, embeddings)
    return embeddings

def cluster_features(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,  # Decreased from 100
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.2,
):
    """
    Performs improved clustering using HDBSCAN.
    
    Args:
        embeddings: UMAP embeddings to cluster
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples for core points
        cluster_selection_epsilon: Distance threshold for cluster membership
    """
    print("Clustering embeddings of shape:", embeddings.shape)

    # Using cuML's HDBSCAN instead of DBSCAN
    clusterer = cuml.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
        gen_min_span_tree=True,
        prediction_data=True
    )

    cluster_labels = clusterer.fit_predict(embeddings)
    cluster_labels = cp.asnumpy(cluster_labels)
    
    # Calculate clustering metrics
    if len(np.unique(cluster_labels)) > 1:  # Only if we have actual clusters
        silhouette_avg = sklearn.metrics.silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
    print(f"Number of clusters found: {len(np.unique(cluster_labels))}")
    print(f"Number of noise points: {np.sum(cluster_labels == -1)}")
    
    return cluster_labels, clusterer

def visualize_clusters(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    output_file: str = 'cluster_space.png',
    figsize: tuple = (12, 8)
):
    """
    Creates an improved visualization of the clustered space.
    """
    plt.figure(figsize=figsize)
    
    # Create a custom colormap with distinct colors for clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Plot noise points first (if any)
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        plt.scatter(
            embeddings[noise_mask, 0],
            embeddings[noise_mask, 1],
            c='lightgray',
            s=5,
            alpha=0.5,
            label='Noise'
        )
    
    # Plot clusters
    for label, color in zip(unique_labels[unique_labels != -1], colors):
        mask = cluster_labels == label
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[color],
            s=20,
            alpha=0.7,
            label=f'Cluster {label}'
        )
    
    plt.title("SAE Feature Space (Colored by Cluster)")
    plt.colorbar()
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center left')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def visualize_clusters_3d(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    output_file: str = 'cluster_space_3d.png',
    figsize: tuple = (12, 8)
):
    """
    Creates a 3D visualization of the clustered space.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a custom colormap with distinct colors for clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Plot noise points first (if any)
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        ax.scatter(
            embeddings[noise_mask, 0],
            embeddings[noise_mask, 1],
            embeddings[noise_mask, 2],
            c='lightgray',
            s=5,
            alpha=0.5,
            label='Noise'
        )
    
    # Plot clusters
    for label, color in zip(unique_labels[unique_labels != -1], colors):
        mask = cluster_labels == label
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            embeddings[mask, 2],
            c=[color],
            s=20,
            alpha=0.7,
            label=f'Cluster {label}'
        )
    
    ax.set_title("SAE Feature Space (3D)")
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def create_interactive_3d_plot(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    patient_ids: np.ndarray,  # Add patient_ids parameter
    output_file: str = 'umap_3d_plot.html',
    sample_size: int = None
):
    """
    Creates an interactive 3D plot using Plotly and saves it as an HTML file.
    
    Args:
        embeddings: UMAP embeddings with shape (n_samples, 3)
        cluster_labels: Cluster assignments
        output_file: Path to save the HTML file
        sample_size: Number of points to sample (optional, for large datasets)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    # Sample points if specified
    if sample_size and sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        cluster_labels = cluster_labels[indices]
        patient_ids = patient_ids[indices]

    # Create a colormap
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)
    colormap = plt.cm.nipy_spectral(np.linspace(0, 1, num_clusters))

    # Create figure
    fig = go.Figure()

    # Add traces for each cluster with hover text
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        color = f'rgb({",".join(map(str, (colormap[i][:3] * 255).astype(int)))})'
        
        name = 'Noise' if label == -1 else f'Cluster {label}'
        marker_size = 3 if label == -1 else 5
        opacity = 0.5 if label == -1 else 0.7

        # Add hover text with patient IDs
        hover_text = [f"PID: {pid}<br>Cluster: {name}" for pid in np.array(patient_ids)[mask]]

        fig.add_trace(go.Scatter3d(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            z=embeddings[mask, 2],
            mode='markers',
            name=name,
            text=hover_text,
            hoverinfo='text',
            marker=dict(
                size=marker_size,
                color=color,
                opacity=opacity,
                line=dict(width=0)
            ),
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text='UMAP 3D Visualization of Feature Space',
            x=0.5,
            y=0.95,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            itemsizing='constant',
            title=dict(text='Clusters'),
            x=1.0,
            y=0.5
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_aspectmode='cube',
        template='plotly_white',
        width=1200,
        height=800,
        showlegend=True
    )

    # Add buttons for different views
    views = {
        'Default View': dict(x=1.5, y=1.5, z=1.5),
        'Top View': dict(x=0, y=0, z=2.5),
        'Side View 1': dict(x=2.5, y=0, z=0),
        'Side View 2': dict(x=0, y=2.5, z=0)
    }

    camera_buttons = []
    for view_name, eye in views.items():
        camera_buttons.append(dict(
            args=[{
                'scene.camera': dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=eye
                )
            }],
            label=view_name,
            method='relayout'
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=camera_buttons,
            x=0.9,
            y=1.1,
            xanchor='right',
            yanchor='top'
        )]
    )

    # Add annotations for controls
    fig.add_annotation(
        text='Controls:<br>- Left click + drag: Rotate<br>- Right click + drag: Pan<br>- Mouse wheel: Zoom',
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.1,
        bordercolor='black',
        borderwidth=1,
        bgcolor='white',
        opacity=0.8
    )

    # Save the figure
    fig.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
        include_mathjax=False
    )
    
    print(f"Interactive 3D plot saved to {output_file}")
    return fig

# Add this after clustering is done
def save_clustering_results(patient_ids, cluster_labels, output_file='cluster_assignments.csv'):
    import pandas as pd
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'cluster': cluster_labels
    })
    
    df.to_csv(output_file, index=False)
    print(f"Saved clustering results to {output_file}")


if __name__ == "__main__":
    # Configuration
    config = {
        'umap__neighbors': 30,
        'umap__min_dist': 0.0,
        'umap__spread': 1.0,
        'umap__num_epochs': 500,
        'umap__n_components': 3,
        'cluster__min_cluster_size': 50,
        'cluster__min_samples': 5,
        'cluster__cluster_selection_epsilon': 0.2,
    }
    sae_dir = './checkpoints/gigapath_sae_wsi2k'
    features_file = './scripts/Gigapath_embeddings.safetensors'
    pid_file = './scripts/Gigapath_ids.txt'
    umap_embeddings_file = 'umap_embeddings.npy'
    cluster_labels_file = 'cluster_labels.npy'

    # Load model and data
    sae = TopkSparseAutoencoder.from_pretrained(sae_dir)
    with open(pid_file, 'r') as f:
        patient_ids = [line.strip() for line in f]
        patient_ids = np.array(patient_ids)
    dataset = SafeTensorDataset(Path(features_file), 'vision')
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

    # Perform UMAP dimensionality reduction
    if not Path(umap_embeddings_file).exists():
        umap_embeddings = umap_reduce(
            sae=sae,
            dataloader=dataloader,
            num_samples=num_samples,
            save_file=umap_embeddings_file,
            **{k.replace('umap__', '') : v for k, v in config.items() if k.startswith('umap__')}
        )
    else:
        umap_embeddings = np.load(umap_embeddings_file)

    # Perform clustering
    if not Path(cluster_labels_file).exists():
        cluster_labels, clusterer = cluster_features(
            umap_embeddings,
            **{k.replace('cluster__', ''): v for k, v in config.items() if k.startswith('cluster__')}
        )
        np.save(cluster_labels_file, cluster_labels)
    else:
        cluster_labels = np.load(cluster_labels_file)

    save_clustering_results(patient_ids, cluster_labels)

    # Create and log visualizations
    if config['umap__n_components'] == 3:
        visualize_clusters_3d(umap_embeddings, cluster_labels)
        create_interactive_3d_plot(umap_embeddings, cluster_labels, patient_ids)
    else:
        visualize_clusters(umap_embeddings, cluster_labels)