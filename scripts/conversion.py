import io
from pathlib import Path

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



