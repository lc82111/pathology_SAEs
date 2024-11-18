from autoencoder import TopkSparseAutoencoder

checkpoints = [
    "cc3m-text-topk-lr-3e-4-k-4-expansion-4",
    "cc3m-text-topk-lr-3e-4-k-16-expansion-4",
    "cc3m-text-topk-lr-3e-4-k-64-expansion-4",
    "cc3m-text-topk-lr-3e-4-k-128-expansion-4",
    "cc3m-vision-topk-lr-3e-4-k-4-expansion-4",
    "cc3m-vision-topk-lr-3e-4-k-16-expansion-4",
    "cc3m-vision-topk-lr-3e-4-k-64-expansion-4",
    "cc3m-vision-topk-lr-3e-4-k-128-expansion-4",
    "cc3m-single_transformer_blocks.9",
    "cc3m-single_transformer_blocks.37",
    "cc3m-transformer_blocks.0-0",
    "cc3m-transformer_blocks.0-1",
    "cc3m-transformer_blocks.18-0",
    "cc3m-transformer_blocks.18-1",
]

for name in checkpoints:
    model = TopkSparseAutoencoder.from_pretrained(f"viccpoes/{name}")
    model.push_to_hub(name)