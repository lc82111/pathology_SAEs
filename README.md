# Sparse Autoencoders for flow matching models

Sparse Autoencoder is a promising technique for interpreting generative model's internal mechanism.
This repository contains training and modeling code which I've used to train SAEs on CLIP and FLUX using CC3M dataset.
For more detailed discussion of results, feel free to check my [writeup](https://re-n-y.github.io/devlog/rambling/sae/).
Check out [fluxlens](https://fluxlens.vercel.app/) for exploring CC3M dataset and SAE features.

This is a work in progress. Feel free to open an issue or rearch out to me via s@krea.ai or @sleenyre on the website formerly represnted by a little blue bird.

## Code structure

1. `autoencoder.py`: Contains modeling code for Topk, Standard, JumpReLU, and Gated SAEs. It also includes training SAE on sampled activations.
2. `fluxsae.py`: Contains SAE training code on FLUX activations. Unlike CLIP, the activations are sampled from flux.schnell on the fly.
3. `notebooks/saevis.ipynb` : Contains minimal code to fetch topk activating samples for each SAE feature.
4. `scripts/conversion.py`: Contains script to download CC3M dataset from HF and load images. The script also provides utils for extracting CLIP activations and saving them to safetensors format.

## Quickstart

```python
from PIL import Image
from transformers import CLIPVisualModel, AutoProcessor
from autoencoder import TopkSparseAutoencoder

sae = TopkSparseAutoencoder.from_pretrained("cc3m-text-topk-lr-3e-4-k-4-expansion-4")
vision = CLIPVisualModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

image = Image.open("cat.jpg")

vision_outputs = vision(pixel_values=batch["pixel_values"], output_hidden_states=True)
acts = vision_outputs.hidden_states[-2][:, 0, :]

# hidden states from SAE only topk activations are kept
hiddens = sae.encode(acts)

# decode the hidden states back to original activations
reconstructed = sae.decode(hiddens)

# increase 12th activation by 42 and return the reconstructed activations
botched = sae.surgery(acts, k=12 strength=42.) 
```

## Why orthnormal constraints and initialisation?

Here are a few interesting facts from [linear algebra](https://en.wikipedia.org/wiki/Orthonormal_basis):

1. An inverse of an orthonormal matrix is its transpose. (i.e. $A^{-1} = A^T$)
2. Given a set of orthonormal basis, the projection of a vector $v$ onto the basis is given by $\langle v, b_i \rangle$ where $b_i$ is the basis vector.

Above facts justify why it's natural to:

1. Tie encoder and decoder weights in SAE (i.e. $W_{enc} = W_{dec}^T$). We want to the SAE to reconstruct the input well.
2. One can interpret the output of encoder ($W_{enc} \cdot x$) as coordinates of $x$ under orthnormal basis given by encoder matrix rows.
3. Intuitively, orthnormal basis contains no "redundant" information since they are orthogonal to each other.

## Open research questions

1. What are good places to sample activations from on FLUX?
2. Given a decent IP-adapter for FLUX, can we use CLIP SAE to steer the generation?
3. Can SAE be the new LoRA? Does it provide enough controllability in style, content, and composition?

## Unorganised experiment notes and ideas

1. I recommend using Topk SAE in general. It's simple, it's effective. Doesn't introduce sparsity weight hparam like other methods
2. Auxiliary loss on Topk SAE is not implemented properly yet.
3. Ghost grads has not been implemented.
4. Encoder / Decoders are initialised to have unit norm columns. Encoder / Decoder weights are tied in init.
5. During training, the decoder columns are normalised to have unit norm.
6. Training SAE on CLIP activation is dirty cheap. On 8xA100 machine, it takes 2~3 minutes to train with standard optimisations.
7. Ideally, if SAE training on flow matching models become reliable in extracting style, content, and composition features, we only have to train one good SAE model and never have to train a LoRA for each model.
8. In some sense, MLPs are already SAEs. For instance, commonly used 4x GeLU MLPs are SAEs if you think about it. GLU is also a gated SAE in some sense.
9. Steering the generation of FLUX using CLIP SAE / FLUX SAE seems ineffective. I've also found that zeroing out CLIP embeddings entirely didn't affect the generation much either. Although, this was done on a handful of examples. I wouldn't bet my money on it.

## Acknowledgements

A lot of the implementations are taken from OpenAI's Topk SAE and dictionary_learning repository. I thank the authors for providing a hackable implementation.
I also thank @enjalot for nudging me to release this implementation.