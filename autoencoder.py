import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoProcessor, AutoTokenizer
from transformers import CLIPVisionModel, CLIPTextModel
from transformers import T5EncoderModel, T5Tokenizer
from transformers import get_constant_schedule_with_warmup
from huggingface_hub import PyTorchModelHubMixin

from datalib import SafeTensorDataset, TextActivationSampler, VisionActivationSampler, TextPoolerActivationSampler

import math
import numpy as np

import wandb
import click
from pathlib import Path
from tqdm import tqdm
from einops import reduce, einsum

# NOTE: implementations are taken from dictionary_learning repository

class TopkSparseAutoencoder(
        nn.Module,
        PyTorchModelHubMixin,
        library_name="finebooru",
        repo_url="https://github.com/RE-N-Y/finebooru"
    ):

    def __init__(self, features: int, pages: int, k: int, sample = None):
        super().__init__()
        self.features = features
        self.pages = pages
        self.k = k

        self.encoder = nn.Linear(features, pages)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(pages, features, bias=False)
        self.bias = nn.Parameter(torch.zeros(features))

        # tie encoder and decoder weights and normalize
        eweight = self.encoder.weight.data.clone()
        self.decoder.weight.data = eweight.T
        self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(dim=0, keepdim=True)

    def encode(self, x, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.bias))
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x):
        return self.decoder(x) + self.bias
    
    def surgery(self, x, k:int, strength:float = 1):
        encoded = self.encode(x)

        # increase the kth feature by strength
        offset = torch.zeros_like(encoded)
        offset[..., k] = strength

        return self.decode(encoded + offset)


    def forward(self, x, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        xhat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return xhat_BD
        else:
            return xhat_BD, encoded_acts_BF
        


# standard sparse autoencoder
class SparseAutoencoder(
        nn.Module,
        PyTorchModelHubMixin, 
        library_name="finebooru",
        repo_url="https://github.com/RE-N-Y/finebooru"
    ):
    def __init__(self, features, pages):
        super().__init__()
        self.features = features
        self.pages = pages
        self.bias = nn.Parameter(torch.zeros(features))
        self.encoder = nn.Linear(features, pages, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(pages, features, bias=False)
        dweight = torch.randn_like(self.decoder.weight)
        dweight = dweight / dweight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dweight)

    # see https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    def encode(self, x):
        f = F.relu(self.encoder(x - self.bias))
        f = f * self.decoder.weight.norm(dim=0, keepdim=True) # normalize with decoder weight norm
        return f
    
    def decode(self, f):
        f = f / self.decoder.weight.norm(dim=0, keepdim=True) # renormalize before decode
        return self.decoder(f) + self.bias
    
    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None: # normal mode
            f = self.encode(x)
            xhat = self.decode(f)
            f = f * self.decoder.weight.norm(dim=0, keepdim=True) # for computing L1 penalty
            if output_features:
                return xhat, f
            else:
                return xhat
        
        else: # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
            f = F.relu(f_pre)

            x_ghost = self.decoder(f_ghost) # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost
            
class GatedAutoEncoder(
        nn.Module,
        PyTorchModelHubMixin, 
        library_name="finebooru",
        repo_url="https://github.com/RE-N-Y/finebooru"
    ):
    """
    An autoencoder with separate gating and magnitude networks.
    """
    def __init__(self, features, pages, device=None):
        super().__init__()
        self.features = features
        self.pages = pages
        self.decoder_bias = nn.Parameter(torch.empty(features, device=device))
        self.encoder = nn.Linear(features, pages, bias=False, device=device)
        self.r_mag = nn.Parameter(torch.empty(pages, device=device))
        self.gate_bias = nn.Parameter(torch.empty(pages, device=device))
        self.mag_bias = nn.Parameter(torch.empty(pages, device=device))
        self.decoder = nn.Linear(pages, features, bias=False, device=device)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        torch.nn.init.zeros_(self.decoder_bias)
        torch.nn.init.zeros_(self.r_mag)
        torch.nn.init.zeros_(self.gate_bias)
        torch.nn.init.zeros_(self.mag_bias)

        # tie encoder and decoder weights and normalize
        eweight = self.encoder.weight.data.clone()
        self.decoder.weight.data = eweight.T
        self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(dim=0, keepdim=True)

    def encode(self, x):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = F.relu(pi_mag)

        f = f_gate * f_mag

        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)
        return f, F.relu(pi_gate)

    def decode(self, f):
        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias
    
    def forward(self, x, output_features=False):
        f, gate = self.encode(x)
        xhat = self.decode(f)

        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            dweight = self.decoder.weight.T.detach()
            bias = self.decoder_bias.detach()
            xhat_gate = gate @ dweight + bias
            return xhat, f, gate, xhat_gate
        else:
            return xhat
        
def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

class StepFunction(torch.autograd.Function):

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold, bandwidth)

    @staticmethod
    def forward(x, threshold, bandwidth):
        y = (x > threshold).to(x.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, threshold, bandwidth = ctx.saved_tensors
        dx = 0.0 * dy
        dthreshold = - (1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * dy
        return dx, dthreshold, None
    
class Step(nn.Module):
    def __init__(self, bandwidth:float):
        super().__init__()
        self.bandwidth = torch.tensor(bandwidth)

    def forward(self, x, threshold):
        return StepFunction.apply(x, threshold, self.bandwidth)

class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold, bandwidth)

    @staticmethod
    def forward(x, threshold, bandwidth):
        y = x * ( x > threshold ).to(x.dtype)
        return y
        

    @staticmethod
    def backward(ctx, dy):
        x, threshold, bandwidth = ctx.saved_tensors
        dx = ( x > threshold ) * dy
        dthreshold = - (threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * dy

        return dx, dthreshold, None
    
class JumpReLU(nn.Module):
    def __init__(self, bandwidth:float):
        super().__init__()
        self.bandwidth = torch.tensor(bandwidth)

    def forward(self, x, threshold):
        return JumpReLUFunction.apply(x, threshold, self.bandwidth)
    

class JumpReLUSparseAutoencoder(
        nn.Module,
        PyTorchModelHubMixin, 
        library_name="finebooru",
        repo_url="https://github.com/RE-N-Y/finebooru"
    ):
    def __init__(self, features:int, pages:int, bandwidth= 0.001, jumpthreshold:float = 0.001):
        super().__init__()
        self.features = features
        self.pages = pages
        self.bias = nn.Parameter(torch.zeros(features))
        self.encoder = nn.Linear(features, pages, bias=True)
        self.decoder = nn.Linear(pages, features, bias=False)
        self.jump = JumpReLU(bandwidth)
        self.step = Step(bandwidth)
        self.logthres = nn.Parameter(torch.full((pages,), math.log(jumpthreshold)))

        # tie encoder and decoder weights and normalize
        eweight = self.encoder.weight.data.clone()
        self.decoder.weight.data = eweight.T
        self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(dim=0, keepdim=True)
        

    def encode(self, x):
        preacts = F.relu(self.encoder(x - self.bias))
        threshold = self.logthres.exp()
        activations = self.jump(preacts, threshold)
        l0 = self.step(activations, threshold).sum(dim=-1)

        return activations, l0
    
    def decode(self, x):
        return self.decoder(x) + self.bias
    
    def forward(self, x, output_features=False):
        f, l0 = self.encode(x)
        xhat = self.decode(f)
        if output_features:
            return xhat, f, l0
        else:
            return xhat


# for warming up the sparsity parameter
class LinearWarmup:
    def __init__(self, num_warmup_steps, lmbda):
        self._step = 0
        self.num_warmup_steps = num_warmup_steps
        self.lmbda = lmbda

    def __call__(self) -> float:
        lmbda = min(self.lmbda, self.lmbda * (self._step / self.num_warmup_steps))
        self._step += 1
        return lmbda        
    
def tohist(data):
    data = data.detach().cpu().numpy()
    data = np.histogram(data, bins=256)
    return wandb.Histogram(np_histogram=data)

class StandardTrainer:
    def __init__(
        self, 
        sae:SparseAutoencoder, 
        optimizer:Optimizer, scheduler:LRScheduler, 
        lmbda:float = 0.5,
        lmbda_warmup_steps:int=256,
        accelerator:Accelerator = None
    ):
        self.sae = sae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lmbda = LinearWarmup(num_warmup_steps=lmbda_warmup_steps, lmbda=lmbda)
        self.accelerator = accelerator
    
    def step(self, x):
        """
        Loss function for a sparse autoencoder.
        x : input to the autoencoder
        """
        self.optimizer.zero_grad()

        b, f = x.shape
        xhat, dictionary = self.sae(x, output_features=True)
        cossim = torch.nn.CosineSimilarity(dim=-1)
        l2 = reduce((x - xhat) ** 2, "b f -> b", "sum").mean()
        l1 = dictionary.norm(p=1, dim=-1).mean()

        l0 = (dictionary > 0).float().sum(dim=-1).mean() # how many entries are alive on average
        alive = (dictionary > 0).float()
        cossim = cossim(x, xhat).mean()

        # Metric taken from OpenAI's TopK SAE paper
        l2_normalised = ((xhat - x) ** 2).mean(dim=-1).mean() / (x ** 2).mean(dim=-1).mean()

        # Feature density (i.e. fraction of samples whose feature is non-zero)
        logdensity = torch.log(reduce(alive, "b f -> f", "mean") + 1e-6) # eps for numerical stability
        # Survival rate (i.e. % of features that are non-zero)
        survival = reduce(dictionary, "b f -> f", "sum") > 0
        survival = survival.float().mean()

        loss = l2 + self.lmbda() * l1

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({
            "loss": loss,
            "l2": l2,
            "l2_normalised": l2_normalised,
            "l1": l1,
            "l0": l0,
            "cossim": cossim,
            "alive": alive.mean(),
            "survival": survival,
            "density": tohist(logdensity)
        })

class TopkTrainer:
    def __init__(
        self, 
        sae:TopkSparseAutoencoder, 
        optimizer:Optimizer, scheduler:LRScheduler, 
        pages:int, auxk:float, bodycount:int,
        normalise:bool = False,
        accelerator:Accelerator = None
    ):
        self.sae = sae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pages = pages
        self.auxk = auxk
        self.bodycount = bodycount
        self.accelerator = accelerator
        self.deaths = torch.zeros(pages, dtype=torch.long, device=accelerator.device)
        self.normalise = normalise

    def step(self, x):
        self.optimizer.zero_grad()

        b, f = x.shape
        xhat, dictionary = self.sae(x, output_features=True)

        cossim = torch.nn.CosineSimilarity(dim=-1)
        l2 = reduce((x - xhat) ** 2, "b f -> b", "sum").mean()
        l1 = dictionary.norm(p=1, dim=-1).mean()

        l0 = (dictionary > 0).float().sum(dim=-1).mean() # how many entries are alive on average
        alive = (dictionary > 0).float()
        cossim = cossim(x, xhat).mean()

        # Metric taken from OpenAI's TopK SAE paper
        l2_normalised = ((xhat - x) ** 2).mean(dim=-1).mean() / (x ** 2).mean(dim=-1).mean()

        # Feature density (i.e. fraction of samples whose feature is non-zero)
        logdensity = torch.log(reduce(alive, "b f -> f", "mean") + 1e-6) # eps for numerical stability
        # Survival rate (i.e. % of features that are non-zero)
        survival = reduce(dictionary, "b f -> f", "sum") > 0
        survival = survival.float().mean()

        loss = l2

        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.sae.parameters(), 1.0)

        if self.normalise:
            self.sae.decoder.weight.data = self.sae.decoder.weight.data / self.sae.decoder.weight.data.norm(dim=0, keepdim=True)
            paralell = einsum(
                self.sae.decoder.weight.grad,
                self.sae.decoder.weight.data,
                'd f, d f -> f'
            )

            self.sae.decoder.weight.grad -= einsum(
                paralell,
                self.sae.decoder.weight.data,
                'f, d f -> d f'
            )

        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({
            "loss": loss,
            "l2": l2,
            "l2_normalised": l2_normalised,
            "l1": l1,
            "l0": l0,
            "cossim": cossim,
            "alive": alive.mean(),
            "survival": survival,
            "density": tohist(logdensity)
        })

class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr, betas=(0, 0.999))
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)

class GatedTrainer:
    def __init__(
        self, 
        sae:GatedAutoEncoder, 
        optimizer:Optimizer, scheduler:LRScheduler, 
        lmbda:float = 0.5,
        lmbda_warmup_steps:int=256,
        accelerator:Accelerator = None
    ):
        self.sae = sae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        
        self.lmbda = LinearWarmup(num_warmup_steps=lmbda_warmup_steps, lmbda=lmbda)

    def step(self, x):
        self.optimizer.zero_grad()

        b, f = x.shape
        xhat, dictionary, gate, xhat_gate = self.sae(x, output_features=True)

        cossim = torch.nn.CosineSimilarity(dim=-1)
        l2 = (x - xhat).pow(2).sum(dim=-1).mean()
        l1 = torch.linalg.norm(gate, ord=1, dim=-1).mean()
        aux = (x - xhat_gate).pow(2).sum(dim=-1).mean()

        l2 = reduce((x - xhat) ** 2, "b f -> b", "sum").mean()
        l1 = dictionary.norm(p=1, dim=-1).mean()
        l0 = (dictionary > 0).float().sum(dim=-1).mean() # how many entries are alive on average
        alive = (dictionary > 0).float()
        
        cossim = cossim(x, xhat).mean()
        l2_normalised = ((xhat - x) ** 2).mean(dim=-1).mean() / (x ** 2).mean(dim=-1).mean()
        survival = reduce(dictionary, "b f -> f", "sum") > 0
        survival = survival.float().mean()
        logdensity = torch.log(reduce(alive, "b f -> f", "mean") + 1e-6) # eps for numerical stability
        loss = l2 + l1 * self.lmbda() + aux

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({
            "loss": loss,
            "aux": aux,
            "l2": l2,
            "l2_normalised": l2_normalised,
            "l1": l1,
            "l0": l0,
            "cossim": cossim,
            "alive": alive.mean(),
            "survival": survival,
            "density": tohist(logdensity)
        })
        

class JumpReLUTrainer:
    def __init__(
        self, 
        sae:JumpReLUSparseAutoencoder, 
        optimizer:Optimizer, scheduler:LRScheduler, 
        lmbda:float = 0.5,
        lmbda_warmup_steps:int=256,
        accelerator:Accelerator = None
    ):
        self.sae = sae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        
        self.lmbda = LinearWarmup(num_warmup_steps=lmbda_warmup_steps, lmbda=lmbda)

    def step(self, x):
        self.optimizer.zero_grad()

        b, f = x.shape
        xhat, dictionary, l0 = self.sae(x, output_features=True)

        cossim = torch.nn.CosineSimilarity(dim=-1)
        l2 = (x - xhat).pow(2).sum(dim=-1).mean()
        l1 = torch.linalg.norm(dictionary, ord=1, dim=-1).mean()

        l2 = reduce((x - xhat) ** 2, "b f -> b", "sum").mean()
        l1 = dictionary.norm(p=1, dim=-1).mean()
        l0 = l0.mean()

        alive = (dictionary > 0).float()
        
        cossim = cossim(x, xhat).mean()
        l2_normalised = ((xhat - x) ** 2).mean(dim=-1).mean() / (x ** 2).mean(dim=-1).mean()
        survival = reduce(dictionary, "b f -> f", "sum") > 0
        survival = survival.float().mean()
        logdensity = torch.log(reduce(alive, "b f -> f", "mean") + 1e-6) # eps for numerical stability
        loss = l2 + l0 * self.lmbda()

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({
            "loss": loss,
            "l2": l2,
            "l2_normalised": l2_normalised,
            "l1": l1,
            "l0": l0,
            "cossim": cossim,
            "alive": alive.mean(),
            "survival": survival,
            "density": tohist(logdensity)
        })
        
def train(**config):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[kwargs])
    accelerator.init_trackers("sae", config=config, init_kwargs={"wandb":{"name":config["name"]}})
    
    match config["modality"], config["model"]:
        case ("text", "openai/clip-vit-base-patch32") | ("text", "openai/clip-vit-large-patch14"):
            tokenizer = AutoTokenizer.from_pretrained(config["model"])
            model = CLIPTextModel.from_pretrained(config["model"], attn_implementation="sdpa")
        case ("vision", "openai/clip-vit-base-patch32") | ("vision", "openai/clip-vit-large-patch14"):
            processor = AutoProcessor.from_pretrained(config["model"])
            model = CLIPVisionModel.from_pretrained(config["model"], attn_implementation="sdpa")
        case "text", "google/t5-v1_1-xxl":
            tokenizer = T5Tokenizer.from_pretrained(config["model"])
            model = T5EncoderModel.from_pretrained(config["model"])
        case _:
            raise ValueError(f"Unknown modality {config['modality']}")

    match config["dataset"]:
        case "tensor":
            dataset = SafeTensorDataset(Path(config["tensorpath"]), config["tensorkey"])
        case _:
            raise ValueError(f"Unknown dataset {config['dataset']} for modality {config['modality']}")   
    
    pages = config["expansion"] * config["features"]

    match config["arch"]:    
        case "standard":
            sae = SparseAutoencoder(features=config["features"], pages=pages)
        case "topk":
            sae = TopkSparseAutoencoder(features=config["features"], pages=pages, k=config["k"])
        case "gated":
            sae = GatedAutoEncoder(features=config["features"], pages=pages)
        case "jumprelu":
            sae = JumpReLUSparseAutoencoder(features=config["features"], pages=pages, bandwidth=config["bandwidth"], jumpthreshold=config["jumpthreshold"])
        case _:
            raise ValueError(f"Unknown architecture {config['arch']}")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["wd"])
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["lr_warmup_steps"])    

    if config["compile"]:
        model = torch.compile(model)
    
    dataloader = dataset.dataloader(batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True)
    sae, optimizer, scheduler, dataloader = accelerator.prepare(sae, optimizer, scheduler, dataloader)
    model = accelerator.prepare(model)

    match config["arch"]:
        case "standard":
            trainer = StandardTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        case "topk":
            trainer = TopkTrainer(sae, optimizer, scheduler, pages=pages, auxk=config["auxk"], bodycount=config["bodycount"], accelerator=accelerator)
        case "gated":
            trainer = GatedTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        case "jumprelu":
            trainer = JumpReLUTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        case _:
            raise ValueError(f"Unknown architecture {config['arch']}, no trainer available")

    match config["sampler"]:
        case "text":
            sampler = TextActivationSampler(model, tokenizer, layer=-2, samples=None)
        case "textpool":
            sampler = TextPoolerActivationSampler(model, tokenizer, samples=None)
        case "vision":
            sampler = VisionActivationSampler(model, processor, layer=-2, samples=None)
        case _:
            raise ValueError(f"Unknown sampler {config['sampler']}")

    for epoch in range(config["num_epochs"]):
        sae.train()
        for batch in tqdm(dataloader):
            x = batch if config["dataset"] == "tensor" else sampler.sample(batch)
            trainer.step(x)

    accelerator.wait_for_everyone()

    if config["savedir"] is not None:
        savedir = Path(config["savedir"]) / config["name"]
        savedir.mkdir(parents=True, exist_ok=True)
        _model = accelerator.unwrap_model(sae)
        _model.save_pretrained(savedir)
    
    accelerator.end_training()

@click.command()
@click.option("--name", type=str, help="Name of the run")
@click.option("--model", type=str, default="openai/clip-vit-base-patch32")
@click.option("--dataset", type=str, default="tensor")
@click.option("--tensorpath", type=str, default=None)
@click.option("--tensorkey", type=str, default=None)
@click.option("--arch", type=str, default="standard")
@click.option("--modality", type=str, default="vision", help="Modality of the dataset. Either 'vision' or 'text'")
@click.option("--sampler", type=str, default="text")
@click.option("--num_workers", type=int, default=96)
@click.option("--num_epochs", type=int, default=10)
@click.option("--batch_size", type=int, default=32)
@click.option("--features", type=int, default=768)
@click.option("--expansion", type=int, default=32)
@click.option("--lmbda", type=float, default=0.01)
@click.option("--lr", type=float, default=5e-5)
@click.option("--beta1", type=float, default=0.9)
@click.option("--beta2", type=float, default=0.999)
@click.option("--wd", type=float, default=0.)
@click.option("--lmbda_warmup_steps", type=int, default=256)
@click.option("--lr_warmup_steps", type=int, default=256)
@click.option("--k", type=int, default=128)
@click.option("--auxk", type=float, default=1/32)
@click.option("--bodycount", type=int, default=16384)
@click.option("--bandwidth", type=float, default=0.001)
@click.option("--jumpthreshold", type=float, default=0.001)
@click.option("--compile", type=bool, default=False)
@click.option("--savedir", type=str, default="./checkpoints")
def main(**config):
    train(**config)


if __name__ == "__main__":
    main()
