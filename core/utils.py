import torch
import torch.nn.functional as F

from typing import Tuple, List

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
    B, C, H, W, E = orig_shape

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

