import torch, torchvision
import numpy as np
from typing import Union, List
import abc
import PIL
import random
from example import Example
"""
https://imgaug.readthedocs.io/en/latest/
"""
import imgaug as ia


class DatasetTransform(abc.ABC):
  @abc.abstractclassmethod
  def __call__(self, example : Example) -> Example:
    pass


class DatasetResize(DatasetTransform):
  _height : int
  _width : int
  _image_transform : torchvision.transforms.Resize

  def __init__(self, height : int, width : int):
    self._height = height
    self._width = width
    self._image_transform = torchvision.transforms.Resize((height, width))

  def __call__(self, example : Example) -> Example:

    image_height = example.image.shape[1]
    image_width = example.image.shape[2]

    transformed_image = self._image_transform(example.image)
    transformed_keypoints = [
      self._scale_keypoint(kp, image_height, image_width) for kp in example.keypoints
    ]

    return Example(transformed_image, transformed_keypoints)

  def _scale_keypoint(self, keypoint : torch.Tensor, image_height, image_width):

    scale = torch.Tensor([float(self._height) / image_height,
                          float(self._width) / image_width])

    return keypoint * scale


class DatasetImageTransform(DatasetTransform):

  def __init__(self, image_transform):
    self._image_transform = image_transform

  def __call__(self, example : Example) -> Example:
    transformed_image = self._image_transform(example.image)
    return Example(transformed_image, example.keypoints)


class ImgaugPipelineTransform(DatasetTransform):

  def __init__(self, imgaug_pipeline):
    self._pipeline = imgaug_pipeline

  def __call__(self, example : Example) -> Example:
    image = np.array(example.image)
    kp_objects = [ia.Keypoint(kp[1], kp[0]) for kp in example.keypoints]
    kpsoi = ia.KeypointsOnImage(kp_objects, image.shape)

    transformed_image, transformed_keypoints = self._pipeline(
      image=image, keypoints=kpsoi
    )

    torch_image = torch.Tensor(transformed_image.copy()).permute((2, 0, 1)) / 255.0
    torch_keypoints = [torch.Tensor([kp.y, kp.x]) for kp in transformed_keypoints.keypoints]

    return Example(image=torch.Tensor(torch_image), keypoints=torch_keypoints)

if __name__ == '__main__':
    example = Example(torch.rand(3, 100, 100), [torch.Tensor([ 25, 50])])
    print(example.keypoints)
    print(DatasetResize(224, 224)(example).keypoints) # expected [tensor([ 56., 112.])]
