# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import torch


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Left-pads a list of 1D tensors with NaNs and stacks them into a 2D tensor.
    Parameters
    ----------
    tensors
        List of 1D tensors.
    Returns
    -------
    torch.Tensor
        2D tensor with NaN-padded 1D tensors.
    """
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)
