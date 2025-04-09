from ruamel.yaml import YAML
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, RandomRotation, Lambda, RandomCrop
from torchvision.transforms.functional import crop
from torchvision.ops import sigmoid_focal_loss

from torch.utils.data import DataLoader, Sampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from functools import partial
from typing import Callable, Tuple, Type, Dict, List, Literal

from utils import DatasetFromTIFF, plot_reconstructs, get_scheduler_with_warmup, plot_markers, PanelBatchSampler
import neptune
from neptune.utils import stringify_unsupported
import matplotlib.pyplot as plt
from math import ceil


from timm.models import VisionTransformer
from modules import CustomConvNeXT, ConvNextBlock, MLP, AttnBlock



class Superkernel(nn.Module):
    def __init__(
            self, 
            num_channels: int,
            embedding_dim: int, 
            num_layers: int,
            num_heads: int,
            mlp_ratio: float,
            layer_type: Literal['conv', 'linear'],
            kernel_size: int = None,
            **kwargs
        ):
        """Initialize the Superkernel model

        Args:
            num_channels (int): Number of channels in the input tensor
            embedding_dim (int): Embedding dimension for the input tensor
            num_layers (int): Number of layers in the model
            num_heads (int): Number of heads per channel embedding
            mlp_ratio (float): MLP ratio for the model
            layer_type (Literal['conv', 'linear']): Whether the output of superkernel should be a convolutional or linear layer weights
            kernel_size (int, optional): Kernel size for the conv layer (already squared). Model embedding will be embedding_dim*kernel_size**2. 
                Number of heads should be a multiplication of squared kernel_size. Defaults to None.
        """
        super(Superkernel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layer_type = layer_type
        self.kernel_size = kernel_size

        model_dim = embedding_dim * kernel_size**2 if layer_type == 'conv' else embedding_dim 
        self.embedder = nn.Embedding(num_channels, model_dim)

        self.encoder = nn.Sequential(*[
            AttnBlock(
                embedding_dim=model_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                mlp_bias=True
            ) 
            for _ in range(num_layers)
        ])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        B, C = indices.shape
        x = self.embedder(indices) # (B, C, E)
        
        if self.num_layers > 0:
            x = self.encoder(x) # (B, C, E)

        if self.layer_type == 'conv':
            # (B, W, C, K, K)
            x = x.reshape(B, C, self.kernel_size, self.kernel_size, self.embedding_dim).permute(0, 4, 1, 2, 3) 
            
        return x


class MultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images."""

    def __init__(
            self,
            encoder_class: Type,
            reshape_fn: Callable = nn.Identity(),
            **kwargs
    ):
        """Initialize the Multiplex Image Encoder.

        Args:
            encoder_class (Type): Encoder class to use.
            reshape_fn (Type, optional): Reshape function to apply to the output of the encoder. Defaults to nn.Identity.
        """
        super().__init__()
        self.encoder = encoder_class(**kwargs)
        self.reshape_fn = reshape_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward_features(x)
        x = self.reshape_fn(x)
        return x


class MultiplexImageDecoder(nn.Module):
    """Decoder for restoring the multiplex image from the embedding tensor."""
    
    def __init__(
            self,
            input_embedding_dim: int,
            decoded_embed_dim: int,
            num_blocks: int,
            scaling_factor: int,
            num_channels: int,
            decoder_layer_type: Type = ConvNextBlock,
            **kwargs
        ) -> None:
            """
            Args:
                input_embedding_dim (int): Embedding dimension of the input tensor.
                decoded_embed_dim (int): Embedding dimension of the decoded tensor (before last projections).
                num_blocks (int): Number of multiplex blocks in each intermediate layer.
                scaling_factor (int): Scaling factor for the upsampling.
                num_channels (int): Number of output channels/markers.
                decoder_layer_type (Type, optional): Type of the decoder layer. Defaults to ConvNextBlock.
                # smooth_conv_channels (int, optional): Number of channels in the smoothing convolution. Defaults to 8.
                # smooth_conv_kernel_size (int, optional): Kernel size of the smoothing convolution. Defaults to 3.
            """
            super().__init__()
            self.scaling_factor = scaling_factor
            self.num_channels = num_channels
            self.decoded_embed_dim = decoded_embed_dim

            self.channel_embed = nn.Embedding(num_channels, input_embedding_dim * decoded_embed_dim) # input projection
            self.channel_biases = nn.Embedding(num_channels, decoded_embed_dim)

            self.decoder = nn.Sequential(*[
                decoder_layer_type(
                    decoded_embed_dim, 
                    **kwargs
                ) for _ in range(num_blocks)
            ])
            self.pred = nn.Conv2d(decoded_embed_dim, scaling_factor**2 * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multiplex Image Decoder.

        Args:
            x (torch.Tensor): Input tensor (embedding).
            indices (torch.Tensor): Indices of the markers.

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        B, I, H, W = x.shape
        E, A = self.decoded_embed_dim, self.scaling_factor

        channel_embeds = self.channel_embed(indices) # [B, C, I*E]
        channel_biases = self.channel_biases(indices) # [B, C, E]
        C = channel_embeds.shape[1]
        N = B * C
        channel_embeds = channel_embeds.reshape(B, C, I, E)
        channel_biases = channel_biases.reshape(B, C, E, 1, 1)

        x = torch.einsum('bihw, bcie -> bcehw', x, channel_embeds)
        x += channel_biases
        x = x.reshape(N, E, H, W)

        x = self.decoder(x)
        x = self.pred(x)

        x = x.reshape(N, A, A, 2, H, W).reshape(B, C, A, A, 2, H, W)
        x = torch.einsum('bcxyohw -> bchxwyo', x)
        # x = x.reshape(N, H*A, W*A).unsqueeze(1)

        # x = self.smooth_conv1(x)
        x = x.reshape(B, C, H*A, W*A, 2)

        return x
    

class MultiplexTransformer(nn.Module):
    """Multiplex image Transformer with Superkernel and Multiplex Image Decoder."""

    def __init__(
            self, 
            num_channels: int,
            input_image_size: int,
            superkernel_embedding_dim: int,
            superkernel_depth: int,
            superkernel_heads: int,
            superkernel_layer_type: Literal['conv', 'linear'],
            encoder_config: Dict,
            decoder_config: Dict,
            superkernel_kernel_size: int = None,
            superkernel_conv_padding: int = None,
            superkernel_conv_stride: int = 1,
            mlp_ratio: float = 4.,
            **kwargs
            ):
        """Initialize the Multiplex Transformer model.

        Args:
            num_channels (int): Number of channels/markers in the dataset.
            input_image_size (int): Size of the input image.
            superkernel_embedding_dim (int): Embedding dimension for the Superkernel.
            superkernel_depth (int): Number of layers in the Superkernel model.
            superkernel_heads (int): Number of heads per channel embedding in the Superkernel model.
            superkernel_layer_type (Literal['conv', 'linear']): Type of the Superkernel layer.
            encoder_config (Dict): Configuration for the encoder.
            decoder_config (Dict): Configuration for the decoder.
            superkernel_kernel_size (int, optional): Size of Superkernel kernel if conv type. Defaults to None.
            superkernel_conv_padding (int, optional): Convolution padding if conv type. Defaults to None.
            superkernel_conv_stride (int, optional): Convolution stride if conv type. Defaults to 1.
            mlp_ratio (float, optional): MLP ratio. Defaults to 4..
        """
        super().__init__()
        self.num_channels = num_channels
        self.input_image_size = input_image_size
        self.superkernel_embedding_dim = superkernel_embedding_dim
        self.superkernel_depth = superkernel_depth
        self.superkernel_heads = superkernel_heads
        self.superkernel_layer_type = superkernel_layer_type
        self.superkernel_kernel_size = superkernel_kernel_size
        self.superkernel_conv_padding = superkernel_conv_padding
        self.superkernel_conv_stride = superkernel_conv_stride
        self.mlp_ratio = mlp_ratio


        self.superkernel = Superkernel(
            num_channels=num_channels, 
            embedding_dim=superkernel_embedding_dim, 
            num_layers=superkernel_depth, 
            num_heads=superkernel_heads, 
            mlp_ratio=mlp_ratio, 
            layer_type=superkernel_layer_type,
            kernel_size=superkernel_kernel_size,
            **kwargs
        )
        self.act = nn.GELU()

        self.encoder = MultiplexImageEncoder(
            **encoder_config
        )

        self.decoder = MultiplexImageDecoder(
            **decoder_config
        )

    def forward(
            self, 
            x: torch.Tensor, 
            encoded_indices: torch.Tensor, 
            decoded_indices: torch.Tensor
        ) -> torch.Tensor:
        B = x.shape[0]
        superkernel_weights = self.superkernel(encoded_indices)
        if self.superkernel_layer_type == 'conv':
            x = torch.cat([
                F.conv2d(
                    x[i].unsqueeze(0), 
                    superkernel_weights[i].to(x.dtype), 
                    padding=self.superkernel_conv_padding,
                    stride=self.superkernel_conv_stride
                )
                for i in range(B)
            ])
            
        else:
            x = torch.einsum('bchw, bce -> behw', x, superkernel_weights.to(x.dtype))
        
        x = self.act(x)
        x = self.encoder(x)
        latent = x
        # latent, features = x[:, 0], x[:, 1:]
        # latent = x.mean(dim=(2, 3))
        # x = features.permute(0, 2, 1).reshape(B, 768, 14, 14)

        x = self.decoder(x, decoded_indices)
        
        return x, latent


# Load the configuration file
config_path = sys.argv[1]
yaml = YAML(typ='safe')
with open(config_path, 'r') as f:
    config = yaml.load(f)


device = config['device']
print(f'Using device: {device}')


TRAIN_DATA_PATH = '/home/duchal/immuvis/imgs/train'
TEST_DATA_PATH = '/home/duchal/immuvis/imgs/test'
SUBDIRS = ['panel1', 'panel2']

TRAIN_PATHS = [f'{TRAIN_DATA_PATH}/{subdir}' for subdir in SUBDIRS]
TEST_PATHS = [f'{TEST_DATA_PATH}/{subdir}' for subdir in SUBDIRS]

SIZE = (112, 112)
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['num_workers']

PANEL_CONFIG = YAML().load(open('configs/panels_config.yaml'))
TOKENIZER = YAML().load(open('configs/markers_tokenizer.yaml'))
INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

CHANNEL_IDS = [
    torch.tensor([TOKENIZER[marker] for marker in panel], dtype=torch.long)
    for panel in PANEL_CONFIG['markers']
]


def plot_reconstructs_with_uncertainty(
        orig_img: torch.Tensor, 
        reconstructed_img: torch.Tensor, 
        sigma_plot: torch.Tensor,
        channel_ids: torch.Tensor,
        masked_ids: torch.Tensor, 
        markers_names_map: Dict[int, str] = INV_TOKENIZER, 
        ncols: int = 9,
        scale_by_max: bool = True,
    ):
    """Plot the original image and the reconstructed image

    Args:
        orig_img (torch.Tensor): Original image
        reconstructed_img (torch.Tensor): Reconstructed image
        sigma_plot (torch.Tensor): Uncertainty image
        channel_ids (torch.Tensor): Indices of the original channels
        masked_ids (torch.Tensor): Indices of the masked/reconstructed channels
        markers_names_map (Dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 8.
        scale_by_max (bool, optional): Whether to scale the images by their maximum value. Defaults to True.

    """
    # plot original image
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / (ncols // 3))
    fig_orig, axs_orig = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    ax_flat = axs_orig.flatten()
    for i in range(0, len(ax_flat), 3):
        j = i // 3

        # first original image
        ax_img = ax_flat[i]
        ax_img.axis('off')

        ax_reconstructed = ax_flat[i+1]
        ax_reconstructed.axis('off')

        ax_uncertainty = ax_flat[i+2]
        ax_uncertainty.axis('off')

        if j < num_channels:
            marker_name = markers_names_map[channel_ids[0, j].item()]
            ax_img.imshow(orig_img[0, j].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=1)
            ax_img.set_title(f'Original\n{marker_name}')

            ax_reconstructed.imshow(reconstructed_img[0, j].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=1)
            is_masked = channel_ids[0, j].item() in masked_ids
            masked_str = ' (masked)' if is_masked else ''
            ax_reconstructed.set_title(f'Reconstructed{masked_str}\n{marker_name}')

            if scale_by_max:
                var_min = sigma_plot[0, j].min().item()
                var_max = sigma_plot[0, j].max().item()
            else:
                var_min = None
                var_max = None

            ax_uncertainty.imshow(sigma_plot[0, j].cpu().numpy(), cmap='CMRmap', vmin=var_min, vmax=var_max)
            ax_uncertainty.set_title(f'Variance\n{marker_name}')
            
    fig_orig.tight_layout()

    return fig_orig


def nll_loss(x, mi, logvar):
    return torch.mean((x - mi)**2 * torch.exp(-logvar) + logvar)

def train_masked(
        model, 
        optimizer,
        scheduler,
        train_dataloader, 
        val_dataloader, 
        device, 
        epochs=10, 
        gradient_accumulation_steps=1,
        min_channels=30,
        run=None
    ):
    """Train a masked autoencoder (decode the remaining channels) with the given parameters."""
    model.train()
    scaler = GradScaler()
    iters = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (img, channel_ids, panel_idx) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):
            batch_size, num_channels, H, W = img.shape

            num_sampled_channels = np.random.randint(min_channels, num_channels)
            channels_subset_idx = [
                np.random.choice(
                    np.arange(num_channels), 
                    size=(1, num_sampled_channels), 
                    replace=False
                ) for _ in range(batch_size)
            ]

            channels_subset_indices = np.concatenate(channels_subset_idx, axis=0)
            channels_subset_indices = torch.tensor(channels_subset_indices, dtype=torch.long)

            channels_subset_indices = channels_subset_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # Shape: [batch_size, num_sampled_channels, H, W]
            masked_img = torch.gather(img, dim=1, index=channels_subset_indices).to(device) 

            # Gather corresponding channel IDs
            active_channel_ids = torch.gather(channel_ids, dim=1, index=channels_subset_indices[:, :, 0, 0]).to(device)


            img = img.to(device)
            channel_ids = channel_ids.to(device)
            masked_img = masked_img.to(torch.float32)

            with autocast(device_type='cuda', dtype=torch.float16):
                output, _ = model(masked_img, active_channel_ids, channel_ids)
                mi, logsigma = output.chunk(2, dim=-1)
                mi = torch.sigmoid(mi.squeeze(-1))
                logsigma = logsigma.squeeze(-1)
                loss = nll_loss(img, mi, logsigma)
            scaler.scale(loss).backward()

            if (batch_idx+1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                run['train/loss'].append(loss.item())
                run['train/lr'].append(scheduler.get_last_lr()[0])
                run['train/µ'].append(mi.mean().item())
                run['train/σ^2'].append(logsigma.mean().item())
                
                iters += 1

            running_loss += loss.item()
                        
        # print(f'Epoch {epoch} loss: {running_loss / len(train_dataloader)}')
        val_loss = test_masked(model, val_dataloader, device, run, epoch, min_channels=min_channels)
        print(f'Validation loss: {val_loss:.4f}')


def test_masked(
        model,  
        test_dataloader, 
        device, 
        run, 
        epoch, 
        num_plots=3, 
        min_channels=20
        ):
    model.eval()
    running_loss = 0.0
    plot_indices = np.random.choice(np.arange(len(test_dataloader)), size=num_plots, replace=False)
    plot_indices = set(plot_indices)
    rand_gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx) in enumerate(test_dataloader):
            batch_size, num_channels, H, W = img.shape
            
            num_sampled_channels = torch.randint(min_channels, num_channels, (1,), generator=rand_gen).item()
            channels_subset_idx = [
                np.random.choice(
                    np.arange(num_channels), 
                    size=(1, num_sampled_channels), 
                    replace=False,
                ) for _ in range(batch_size)
            ]

            channels_subset_indices = np.concatenate(channels_subset_idx, axis=0)
            channels_subset_indices = torch.tensor(channels_subset_indices, dtype=torch.long)
            
            channels_subset_indices = channels_subset_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # Shape: [batch_size, num_sampled_channels, H, W]
            masked_img = torch.gather(img, dim=1, index=channels_subset_indices).to(device) 

            # Gather corresponding channel IDs
            active_channel_ids = torch.gather(channel_ids, dim=1, index=channels_subset_indices[:, :, 0, 0]).to(device)

            channel_ids = channel_ids.to(device)
            masked_img = masked_img.to(torch.float32)
            img = img.to(device)

            output, _ = model(masked_img, active_channel_ids, channel_ids)
            mi, logsigma = output.chunk(2, dim=-1)
            mi = torch.sigmoid(mi.squeeze(-1))
            logsigma = logsigma.squeeze(-1)
            loss = nll_loss(img, mi, logsigma)
            running_loss += loss.item()

            if idx in plot_indices:
                uncetrainty_img = torch.exp(logsigma)
                unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
                masked_channels_names = '\n'.join([INV_TOKENIZER[i.item()] for i in unactive_channels])
                reconstr_img = plot_reconstructs_with_uncertainty(
                    img,
                    mi,
                    uncetrainty_img,
                    channel_ids,
                    unactive_channels,
                    scale_by_max=False
                )
                
                run['val/imgs'].append(reconstr_img, description=f'Resuilting outputs (variance scaled individually each plot) (panel {panel_idx[0].item() +1}, epoch {epoch})'+\
                                       '\n\nMasked channels: {}'.format(masked_channels_names))
                plt.close('all') 

                reconstr_img = plot_reconstructs_with_uncertainty(
                    img,
                    mi,
                    uncetrainty_img,
                    channel_ids,
                    unactive_channels,
                    scale_by_max=True
                )
                run['val/imgs'].append(reconstr_img, description=f'Resuilting outputs (variance scaled by min-max)  (panel {panel_idx[0].item() +1}, epoch {epoch})'\
                                       '\n\nMasked channels: {}'.format(masked_channels_names))
                plt.close('all')
                
    
    val_loss = running_loss / len(test_dataloader)
    run['val/loss'].append(val_loss)
    
    return val_loss





def crop_val(image):
    init_x, init_y = 0, 0
    return crop(image, init_y, init_x, init_y + SIZE[1], init_x + SIZE[0])

train_transform = Compose([
    RandomRotation(180),
    RandomCrop(SIZE),
])

test_transform = Lambda(crop_val)


train_dataset = DatasetFromTIFF(TRAIN_PATHS, CHANNEL_IDS, transform=train_transform)
test_dataset = DatasetFromTIFF(TEST_PATHS, CHANNEL_IDS, transform=test_transform)

train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=NUM_WORKERS)


model_config = {
    'num_channels': len(TOKENIZER),
    **config['superkernel']
}

encoder_config = {
    "encoder_class": CustomConvNeXT,
    **config['encoder']
}

decoder_config = {
    **config['decoder'],
    'num_channels': len(TOKENIZER),
    'decoder_layer_type': ConvNextBlock,
}

model = MultiplexTransformer(**model_config, encoder_config=encoder_config, decoder_config=decoder_config).to(device)


lr = config['lr']
final_lr = config['final_lr']
weight_decay = config['weight_decay']
gradient_accumulation_steps = config['gradient_accumulation_steps']
epochs = config['epochs']
num_warmup_steps = config['num_warmup_steps']
num_annealing_steps = config['num_annealing_steps']

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_scheduler_with_warmup(optimizer, num_warmup_steps, num_annealing_steps, final_lr=final_lr, type='cosine')


run = neptune.init_run(
    project=...,
    api_token=...,
    tags=[tag for tag in config['tags']],
)

run["parameters"] = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "lr": lr,
    "weight_decay": weight_decay,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "epochs": epochs,
    "num_warmup_steps": num_warmup_steps,
    "num_annealing_steps": num_annealing_steps,
    "model_general_config": stringify_unsupported(model_config),
    "encoder_config": stringify_unsupported(encoder_config),
    "decoder_config": stringify_unsupported(decoder_config),
}


train_masked(
    model, 
    optimizer, 
    scheduler,
    train_dataloader, 
    test_dataloader, 
    device, 
    epochs=epochs, 
    gradient_accumulation_steps=gradient_accumulation_steps,
    run=run,
    min_channels=config['min_channels']
)

run_name = run['sys/name'].fetch()
run.stop()

torch.save(model.state_dict(), f'model-{run_name}.pth')
print('Finished training!')

