import os
import random
import numpy as np
import tifffile
import anndata
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, Sampler
from glob import glob
from cv2 import medianBlur
from skimage import filters
from typing import Tuple, List, Dict, Literal
from math import ceil, cos, pi
from functools import partial


class DatasetFromTIFF(Dataset):
    def __init__(
            self, 
            img_paths: List[str],
            channel_ids: List[List[int]],
            transform=None,
            use_median_denoising: bool = False,
            use_butterworth_filter: bool = True

        ):
        assert len(img_paths) == len(channel_ids), 'Each image path has to have a corresponding channel ids'
        self.channel_ids = channel_ids
        self.imgs = [] # tuples of (img_path, channel_ids index)
        for i, img_path in enumerate(img_paths):
            tiffs = glob(os.path.join(img_path, '*.tiff'))
            self.imgs.extend([(tiff, i) for tiff in tiffs])
        
        self.transform = transform
        self.use_denoising = use_median_denoising
        self.use_butterworth = use_butterworth_filter
    
    @staticmethod
    def preprocess(img):
        return np.arcsinh(img / 5.0)
    
    @staticmethod
    def denoise(img):
        denoised_channels = [
            medianBlur(img[i].astype('float32'), 3) 
            for i in range(img.shape[0])
        ]
        return np.stack(denoised_channels)
    
    @staticmethod
    def butterworth(img):
        filtered_channels = [
            filters.butterworth(img[i], cutoff_frequency_ratio=0.2, high_pass=False)
            for i in range(img.shape[0])
        ]
        return np.stack(filtered_channels)

    @staticmethod
    def norm_minmax(img):
        min_val = np.min(img, axis=(1,2), keepdims=True)
        max_val = np.max(img, axis=(1,2), keepdims=True)
        scaled_img = np.where(
            max_val == min_val,
            img,
            (img - min_val) / (max_val - min_val + 1e-8)
        )
        return torch.tensor(scaled_img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, panel_channels_idx = self.imgs[idx]
        img = tifffile.imread(img_path)   
        img = self.preprocess(img)
        if self.transform:
            img = self.transform(torch.tensor(img)).numpy()
        if self.use_butterworth:
            img = self.butterworth(img)
        if self.use_denoising:
            img = self.denoise(img)
        img = self.norm_minmax(img)
        
        channel_ids = self.channel_ids[panel_channels_idx]
        return img, channel_ids, panel_channels_idx


class PanelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by panel
        self.panel_to_indices = {}
        for idx, (_, panel_idx) in enumerate(dataset.imgs):
            if panel_idx not in self.panel_to_indices:
                self.panel_to_indices[panel_idx] = []
            self.panel_to_indices[panel_idx].append(idx)
        
        # Convert to list of (panel, indices) pairs for easier random selection
        self.panels = list(self.panel_to_indices.keys())

        self.epoch_batches = []  # Store batches for an epoch
        self._generate_batches()  # Prepare the first epoch

    def _generate_batches(self):
        """Generate batches ensuring each sample is used exactly once per epoch."""
        self.epoch_batches = []  # Reset batches for the new epoch
        
        # Shuffle panels if needed
        if self.shuffle:
            random.shuffle(self.panels)

        for panel in self.panels:
            indices = self.panel_to_indices[panel]
            
            # Shuffle indices within the panel if needed
            if self.shuffle:
                random.shuffle(indices)

            # Split indices into batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                self.epoch_batches.append(batch)

        # Shuffle the final batch order for diversity
        if self.shuffle:
            random.shuffle(self.epoch_batches)

    def __iter__(self):
        """Yield batches, ensuring all images are used exactly once per epoch."""
        for batch in self.epoch_batches:
            yield batch
        self._generate_batches()  # Prepare for next epoch

    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.epoch_batches)


def plot_markers(image: np.ndarray, marker_names: Dict[int, str], cmap: str = 'CMRmap', save_path: str = None):
    """Plot all the markers in the image 

    Args:
        image (np.ndarray): Image to plot
        marker_names (dict[int:str]): Mapper from marker index to marker name
        cmap (str, optional): Color map to use. Defaults to 'CMRmap'.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    num_cols = 8 if len(marker_names) <= 40 else 9
    fig, axs = plt.subplots(num_cols, 5, figsize=(12, 20))
    for i, ax in enumerate(axs.flat):
        ax.imshow(image[i], cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(marker_names[i])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_reconstructs(
        orig_img: torch.Tensor, 
        reconstructed_img: torch.Tensor, 
        masked_idx: torch.Tensor, 
        markers_names_map: Dict[int, str], 
        ncols: int = 8,
        vmax: int = 1
    ):
    """Plot the original image and the reconstructed image

    Args:
        orig_img (torch.Tensor): Original image
        reconstructed_img (torch.Tensor): Reconstructed image
        masked_idx (torch.Tensor): Indices of the masked/reconstructed channels
        markers_names_map (Dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 8.

    """
    # plot original image
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / ncols)
    fig_orig, axs_orig = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    for i, ax in enumerate(axs_orig.flatten()):
        ax.axis('off')
        if i < num_channels:
            ax.imshow(orig_img[0, i].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=vmax)
            ax.set_title(markers_names_map[i])
            
    fig_orig.tight_layout()

    # plot reconstructed image
    num_channels = reconstructed_img.shape[1]
    nrows = ceil((2 * num_channels) / ncols)
    fig_reconstructed, axs_reconstructed = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    for i, ax in enumerate(axs_reconstructed.flatten()):
        ax.axis('off')
        if i < 2*num_channels:
            channel = masked_idx[0, i // 2].item()
            if i % 2 == 0:
                # plot original image on even indices
                ax.imshow(orig_img[0, channel].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=vmax)
            else:
                ax.imshow(reconstructed_img[0, i//2].cpu().numpy(), cmap='CMRmap', vmin=0, vmax=vmax)
            ax.set_title(markers_names_map[channel])
            
    fig_reconstructed.tight_layout()
    return fig_orig, fig_reconstructed


def get_marker_names_map(adata: anndata.AnnData) -> Dict[int, str]:
    """Get the marker names from the anndata object

    Args:
        adata (anndata.AnnData): Anndata object

    Returns:
        dict[int:str]: Mapper from marker index to marker name
    """
    df = adata.var.copy()
    df.index = df.index.astype(int)
    return df['marker'].to_dict()


def get_scheduler_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_annealing_steps: int,
        final_lr: float,
        type: Literal['cosine', 'linear'] = 'cosine'
    ) -> LambdaLR:
    """Get cosine annealing scheduler with warmup, adapted from:
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        num_warmup_steps (int): Number of warmup steps
        num_annealing_steps (int): Number of cosine annealing steps
        final_lr (float): Minimum learning rate after annealing

    Returns:
        LambdaLR: Scheduler
    """
    def lr_lambda(current_step, type: Literal['cosine', 'linear'] = 'cosine'):
        if current_step < num_warmup_steps:
            return float(max(1, current_step)) / float(max(1, num_warmup_steps))
        elif current_step >= num_annealing_steps + num_warmup_steps:
            return final_lr
        
        progress = (current_step - num_warmup_steps) / float(max(1, num_annealing_steps - num_warmup_steps))
        
        if type == 'linear':
            return max(final_lr, (1.0 - progress) * (1.0 - final_lr) + final_lr)
        
        return final_lr + (1.0 - final_lr) * 0.5 * (1.0 + cos(pi * progress))

    lr_lambda = partial(lr_lambda, type=type)
    return LambdaLR(optimizer, lr_lambda, -1)
