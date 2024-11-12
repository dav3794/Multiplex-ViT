import os
import numpy as np
import tifffile
import anndata
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torchvision.transforms import RandomResizedCrop
from torch.utils.data import Dataset
from glob import glob
from typing import Tuple, List


class DatasetFromTIFF(Dataset):
    def __init__(self, img_path: str, transform=None):
        self.imgs = glob(os.path.join(img_path, '*.tiff'))
        self.transform = transform
    
    @staticmethod
    def preprocess(img):
        return np.arcsinh(img / 5.0)
    
    @staticmethod
    def norm_minmax(img):
        min_val = np.min(img, axis=(1,2), keepdims=True)
        max_val = np.max(img, axis=(1,2), keepdims=True)
        scaled_img = (img - min_val) / (max_val - min_val)
        return torch.tensor(scaled_img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = tifffile.imread(self.imgs[idx])
        img = self.preprocess(img)
        img = self.norm_minmax(img)
        if self.transform:
            img = self.transform(img)
        channel_ids = torch.arange(img.shape[0], dtype=torch.long)
        return img, channel_ids


def plot_markers(image: np.ndarray, marker_names: dict[int:str], cmap: str = 'CMRmap', save_path: str = None):
    """Plot all the markers in the image 

    Args:
        image (np.ndarray): Image to plot
        marker_names (dict[int:str]): Mapper from marker index to marker name
        cmap (str, optional): Color map to use. Defaults to 'CMRmap'.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig, axs = plt.subplots(8, 5, figsize=(12, 20))
    for i, ax in enumerate(axs.flat):
        ax.imshow(image[i], cmap=cmap)
        ax.axis('off')
        ax.set_title(marker_names[i])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_marker_names_map(adata: anndata.AnnData) -> dict[int:str]:
    """Get the marker names from the anndata object

    Args:
        adata (anndata.AnnData): Anndata object

    Returns:
        dict[int:str]: Mapper from marker index to marker name
    """
    df = adata.var.copy()
    df.index = df.index.astype(int)
    return df['marker'].to_dict()

# code adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
# and https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py

def window_partition(x: torch.Tensor, window_size: int, shift: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
        shift (Tuple): shift in y and x pixels (from top left corner to bottom & right) for windows.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition.
    """
    if shift[0] > 0 or shift[1] > 0:
        x = F.pad(x, (0, 0, shift[1], 0, shift[0], 0))

    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, 
    window_size: int, 
    shift: Tuple[int, int], 
    pad_hw: Tuple[int, int], 
    orig_shape: List[int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        shift (Tuple): shift in y and x pixels (from top left corner to bottom & right) for windows.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        orig_shape (List): original shape before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    shift_y, shift_x = shift
    Hp, Wp = pad_hw
    B, H, W, C = orig_shape

    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if shift_y > 0 or shift_x > 0:
        x = x[:, shift_y:, shift_x:, :].contiguous()
    
    if Hp - shift_y > H or Wp - shift_x > W:
        x = x[:, :H, :W, :].contiguous()

    return x


def patch_partition(x: torch.Tensor, patch_size: int, shift: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping flattened patches with padding if needed.
    Args:
        x (tensor): input tokens with [B, C, H, W, E].
        patch_size (int): patch size.
        shift (Tuple): shift in y and x pixels (from top left corner to bottom & right) for patches.

    Returns:
        patches: patches after partition with [B * num_patches, C, patch_size*patch_size, E].
        (Hp, Wp): padded height and width before partition.
    """
    if shift[0] > 0 or shift[1] > 0:
        x = F.pad(x, (0, 0, shift[1], 0, shift[0], 0))

    B, C, H, W, E = x.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, C, Hp // patch_size, patch_size, Wp // patch_size, patch_size, E)
    x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, C, patch_size * patch_size, E) # B * num_patches, C, patch_size*patch_size, E

    return x, (Hp, Wp)


def patch_unpartition(
    patches: torch.Tensor, 
    patch_size: int, 
    shift: Tuple[int, int],
    pad_hw: Tuple[int, int], 
    orig_shape: Tuple[int, int, int, int, int]
) -> torch.Tensor:
    """
    Patch unpartition into original sequences and removing padding.
    Args:
        patches (tensor): input tokens with [B * num_patches, C, patch_size*patch_size, E].
        patch_size (int): patch size.
        shift (Tuple): shift in y and x pixels (from top left corner to bottom & right) for patches.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        orig_shape (Tuple): original shape of tensor, before padding.

    Returns:
        x: unpartitioned sequences with [B, C, H, W, E].
    """
    shift_y, shift_x = shift
    Hp, Wp = pad_hw
    B, C, H, W, E = orig_shape
    x = patches.view(B, Hp // patch_size, Wp // patch_size, C, patch_size, patch_size, E)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, C, Hp, Wp, E)

    if shift_y > 0 or shift_x > 0:
        x = x[:, shift_y:, shift_x:, :].contiguous()
    
    if Hp - shift_y > H or Wp - shift_x > W:
        x = x[:, :H, :W, :].contiguous()
    return x

