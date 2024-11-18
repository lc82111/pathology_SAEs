from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from PIL.Image import Image as PILImage
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class SafeTensorDataset(Dataset):
    def __init__(self, path:Path, key:str):
        with safe_open(path, framework="pt") as f:
            self.data = f.get_tensor(key)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
    
class TextActivationSampler:
    def __init__(self, model:AutoModel, tokenizer:AutoTokenizer, layer:int=-2, samples:int|None=None):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.samples = samples

    @torch.no_grad()
    def sample(self, batch:dict[str,list[str]]):
        texts = batch["captions"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        masks = rearrange(inputs["attention_mask"], "b t -> (b t)")
        outputs = self.model(**inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[self.layer]
        outputs = rearrange(outputs, "b t d -> (b t) d")
        outputs = outputs[torch.nonzero(masks)] # don't sample from MASK tokens!
        
        bt, d = outputs.shape
        outputs = outputs[torch.randperm(bt)]

        bsz = self.samples or bt
        return outputs[:bsz]

class VisionActivationSampler:
    def __init__(self, model:AutoModel, processor:AutoProcessor, layer:int=-2, samples:int|None=None):
        self.model = model
        self.processor = processor
        self.layer = layer
        self.samples = samples

    @torch.no_grad()
    def sample(self, batch:dict[str,list[PILImage]]):
        images = batch["images"]
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[self.layer][:, 0, :] # sample CLS token

        return outputs
    
class TextPoolerActivationSampler:
    def __init__(self, model:AutoModel, tokenizer:AutoTokenizer, layer:int=-2, samples:int|None=None):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.samples = samples

    @torch.no_grad()
    def sample(self, batch:dict[str,list[str]]):
        texts = batch["captions"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        outputs = outputs.pooler_output

        return outputs
    

class Hook:
    def __init__(self, model:nn.Module, location:str, hook):
        self.model = model
        self.location = location
        self.module = self.model.get_submodule(location)

        self.cached = None # cache the last output
        def _hook(module, args, output):
            self.cached = output
            return hook(module, args, output)

        self.module.register_forward_hook(_hook)


    def run(self, *args, **kwargs):
        return self.model(*args, **kwargs)
