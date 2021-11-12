import torch
import PIL
from typing import Union, List

class Example:
  image : Union[torch.Tensor, PIL.Image.Image]
  keypoints : List[torch.Tensor]

  def __init__(self, image, keypoints):
    self.image = image
    self.keypoints = keypoints