import io
import json
from pathlib import Path
import pandas as pd
import numpy as np
import h5py

import click
from PIL import Image
from tqdm import tqdm
import polars as pl

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel

from accelerate import Accelerator
from huggingface_hub import HfFileSystem
from safetensors.torch import save_file


class CC3M(Dataset):
    def __init__(self, path:Path):
        data = []
        for file in path.glob("*.parquet"):
            data.append(pl.read_parquet(file))
        self.dataset = pl.concat(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.row(idx, named=True)
        image = Image.open(io.BytesIO(row['image']['bytes']))
        text = row["conversations"][-1]['value']
        return image, text


@click.group()
def cli():
    pass

@cli.command("create_intersection_between_h5_png")
def create_intersection_between_h5_png(h5_path, png_path):
    base_dir = Path("/mnt/nvme0n1p1/datasets/patholog/FD_data")

    # png files on disk
    json_file = base_dir / 'FDMSI_patches_359277_slides_484.json'
    png_paths = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for pid, coords in data.items():  # {'WSI path': [tile1_name, ...]}
            for coords in coords:
                png_paths.append(f"{pid}/{coords}")

    # h5 files on disk
    gigapath_feat_dir = base_dir/'Features/Gigapath' 
    h5_paths = [f.stem for f in gigapath_feat_dir.glob('*.h5')]    

    found_list, miss_list = [], []
    for pid in tqdm(h5_paths, desc="find PID/coords in h5"):
        with h5py.File(pid, 'r') as f:
            coords = f['coords'][:]
            coords = [coord.decode('utf-8') for coord in coords]
            for coord in coords:
                h5_path = f"{pid}/{coord}"
                for png_path in png_paths:
                    pair = {'h5_path':h5_path, 'png_path':png_path}
                    if pid in png_path and coord in png_path:
                        found_list.append(pair)
                    else:
                        miss_list.append(pair)

    # save output to json
    with open('h5_png_intersection_found.json', 'w') as f:
        json.dump(found_list, f)
    with open('h5_png_intersection_miss.json', 'w') as f:
        json.dump(miss_list, f)
    print('done')


def get_dinov2_feature(data_root_dir="/mnt/nvme0n1p1/datasets/patholog/FD_data/Features/", dinov2_model_name='Gigapath'): 
    '''Return all features, labels and patient ID for all patients'''
    data_root_dir = Path(data_root_dir)

    def _load_feats(root_dir, df):
        """Load features and coords from h5 files for all patients in the dataframe"""

        # h5 files on disk 
        feats_list, files_list = [], []
        for pid in tqdm(df['PATIENT'], desc="Loading features"):
            path = Path(root_dir) / f"{pid}.h5"
            with h5py.File(path, 'r') as f:
                feats = f['feats'][:]
                feats_list.append(feats)

                coords = f['coords'][:]
                coords = [coord.decode('utf-8') for coord in coords]
                full_paths = [f"{pid}/{coord}" for coord in coords]
                files_list.append(full_paths)
        return feats_list, files_list
   
    def _load_and_process(df):
        feats, files = _load_feats(data_root_dir / dinov2_model_name, df)  # (n_patients, n_tiles, n_features), (n_patients, n_tiles)

        labels = df['MSIStatus'].values # (n_patients,)
        labels = np.repeat(labels, [feat.shape[0] for feat in feats], axis=0) # (n_patients, n_tiles) 
        labels = labels.flatten() # (n_samples,)

        feats = np.vstack([feat for feat in feats]) # reshape to (n_samples, n_features)
        files = np.concatenate(files, axis=0).tolist() # (n_samples,)
        return feats, labels, files

    train_df = pd.read_csv(data_root_dir / 'train.csv')[['PATIENT', 'MSIStatus']]
    val_df = pd.read_csv(data_root_dir / 'valid.csv')[['PATIENT', 'MSIStatus']]
    test_df = pd.read_csv(data_root_dir / 'test.csv')[['PATIENT', 'MSIStatus']]

    train_feats, train_labels, train_files = _load_and_process(train_df)
    val_feats, val_labels, val_files = _load_and_process(val_df)
    test_feats, test_labels, test_files = _load_and_process(test_df)

    X = np.concatenate([train_feats, val_feats, test_feats], axis=0) # (n_samples, n_features)
    y = np.concatenate([train_labels, val_labels, test_labels], axis=0) # (n_samples,)
    files = train_files + val_files + test_files # (n_samples,)

    return torch.tensor(X), torch.tensor(y), files

@cli.command("write_dinov2")
def write_dinov2(dinov2_model_name='Gigapath'):
    X, y, fns = get_dinov2_feature(dinov2_model_name=dinov2_model_name) 

    # (n_samples, n_features), (n_samples,), (n_samples,)
    print(f"X: {X.shape}, y: {y.shape}, file names: {len(fns)}")

    # save embeddings to a safetensor file
    save_file(tensors={ "vision": X, "label": y}, filename=f"./{dinov2_model_name}_embeddings.safetensors")
    print(f"Saved {dinov2_model_name} embeddings to {dinov2_model_name}_embeddings.safetensors")

    # save ids to a txt file
    with open(f"{dinov2_model_name}_ids.txt", "w") as f:
        f.write("\n".join(fns))
    print(f"Saved {dinov2_model_name} ids to {dinov2_model_name}_ids.txt")

# for generating clip embeddings on cc3m and saving them to safetensor file
@cli.command("embed")
def embed():
    accelerator = Accelerator(mixed_precision="bf16")

    # Load the processor and the model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", attn_implementation="sdpa")
    text = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", attn_implementation="sdpa")
    cc3m = CC3M(Path("/data/cc3m/"))

    def collate_fn(batch):
        images = [im for im, _ in batch]
        texts = [txt for _, txt in batch]

        return processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True)

    dataloader = torch.utils.data.DataLoader(cc3m, batch_size=1024, collate_fn=collate_fn)
    vision = torch.compile(vision)
    text = torch.compile(text)
    vision, text, dataloader = accelerator.prepare(vision, text, dataloader)

    tembeds = []
    vembeds = []

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing embeddings"):
        with torch.no_grad():
            text_outputs = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            text_outputs = text_outputs.pooler_output.cpu()
            vision_outputs = vision(pixel_values=batch["pixel_values"], output_hidden_states=True)
            vision_outputs = vision_outputs.hidden_states[-2][:, 0, :].cpu()

        tembeds.append(text_outputs)
        vembeds.append(vision_outputs)

    # Save the embeddings to a file
    save_file({ "vision": torch.cat(vembeds), "text": torch.cat(tembeds),}, "/data/cc3m/embeddings.safetensors")  

# for writing images to disk
@cli.command("write")
def write():
    cc3m = CC3M(Path("/data/cc3m/"))
    outdir = Path("/data/cc3m/images/")
    outdir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(cc3m)), desc="Writing images"):
        image, text = cc3m[idx]
        image.save(outdir / f"{idx}.jpg")


# for downloading LLaVA recap cc3m dataset
@cli.command("download")
def download():
    fs = HfFileSystem()
    outdir = Path("/data/cc3m")
    files = fs.glob("datasets/lmms-lab/LLaVA-ReCap-CC3M/data/*.parquet")

    for file in tqdm(files):
        with fs.open(file, "rb") as r:
            file = Path(file)
            with open(outdir / file.name, "wb") as w:
                w.write(r.read()) 

if __name__ == "__main__":
    cli()



