import torch, torchvision
import Enum
import pathlib
import augment
from example import Example
import pandas as pd
from typing import List, Tuple, Dict, Union, String
import numpy as np
import PIL
"""
https://imgaug.readthedocs.io/en/latest/
"""
import imgaug.augmenters as iaa

class KeyPointsBatch(Enum):
  part_detection = 1
  location_refinement_map = 2
  location_refinement_mask = 3
  keypoint = 4

def get_key_points_dataset(path : String, labels_file : String, transform = torch.nn.Identity()):

  labels = pd.read_csv(labels_file, header=None)
  #nose_points = {'nose': [(labels[1][i], labels[2][i]) for i in range(len(labels))]}

  nose_points = [[torch.Tensor([labels[2][i], labels[1][i]])] for i in range(len(labels))]
  return KeyPointsDataset([pathlib.Path(path) / f for f in labels[3]], nose_points, transform)

class KeyPointsDataset(torch.utils.data.Dataset):
  files : List[pathlib.Path]
  keypoints : List[List[torch.Tensor]] # Each image has a list of torch.Tensor keypoints. Each keypoint is 2 values.
  _width : int = 129
  _height : int = 323
  _scmap_width : int = 18
  _scmap_height : int = 42
  score_map_gaussian_std : float
  locref_scale : float

  def __init__(self, files, keypoints, transform, score_map_gaussian_std, locref_scale):
    self.files = files
    self.keypoints = keypoints
    self._transform = transform

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[KeyPointsBatch, torch.Tensor]]:

    image = self._load_image(idx)
    example = Example(image, self.keypoints[idx])

    transformed_example = self._transform(example)

    keypoint_maps = [
      self._compute_score_map(k) for k in transformed_example.keypoints
    ]

    output = {
        KeyPointsBatch.part_detection: keypoint_maps[0][0],
        KeyPointsBatch.location_refinement_map: keypoint_maps[0][1],
        KeyPointsBatch.location_refinement_mask: keypoint_maps[0][2],
        KeyPointsBatch.keypoint: transformed_example.keypoints[0]
    }

    return transformed_example.image, output

  def _load_image(self, idx):
    return PIL.Image.open(str(self.files[idx]))

  def _scale_keypoint(self, keypoint, image_width, image_height):
    return (int(keypoint[0] / image_width * self._width), int(keypoint[1] / image_height * self._height))

  def _keypoint_to_pixels(self, keypoint):
    pixels = torch.zeros((1, self._scmap_height, self._scmap_width))
    pixels[0, keypoint[1], keypoint[0]] = 1
    return pixels

  def _compute_score_map(self, keypoint):

    stride = torch.Tensor([self._height / self._scmap_height, self._width / self._scmap_width])

    coords = torch.from_numpy(np.mgrid[:self._scmap_height, :self._scmap_width])
    coords[0, ...] = coords[0]*stride[0] + stride[0] / 2
    coords[1, ...] = coords[1]*stride[1] + stride[1] / 2

    location_refinement_map = torch.zeros((self._scmap_height, self._scmap_width, 2))
    location_refinement_mask = torch.zeros((self._scmap_height, self._scmap_width, 2))
    dist_thresh = 2*self.score_map_gaussian_std
    dist_thresh_sq = dist_thresh ** 2

    dw = keypoint[1] - coords[1]
    dh = keypoint[0] - coords[0]

    square_distance = dw**2 + dh**2
    pixels = torch.exp(-square_distance/(2*self.score_map_gaussian_std**2))

    location_refinement_mask[..., 0] = (square_distance <= dist_thresh_sq).float()
    location_refinement_mask[..., 1] = (square_distance <= dist_thresh_sq).float()
    location_refinement_map[..., 0] = dh * self.locref_scale * location_refinement_mask[..., 0]
    location_refinement_map[..., 1] = dw * self.locref_scale * location_refinement_mask[..., 1]

    # Add a first dimension to represent the different joints
    return pixels.unsqueeze(0), location_refinement_map.unsqueeze(0), location_refinement_mask.unsqueeze(0)


def prepare_train_valid_loaders(trainset, valid_fraction=0.1, batch_size=2):
    '''
    Split trainset data and prepare DataLoader for training and validation

    Args:
        trainset (Dataset): data
        valid_size (float): validation size
        batch_size (int) : batch size
    '''

    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_fraction * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

if __name__ == '__main__':
    image, sample_output = kp_dataset[0]
    #show(image)
    scmap = sample_output[KeyPointsBatch.part_detection]
    locref_map = sample_output[KeyPointsBatch.location_refinement_map]
    locref_mask = sample_output[KeyPointsBatch.location_refinement_mask]
    #show(scmap)
    #show([locref_map[..., 0], locref_map[..., 1], locref_mask[..., 0]])
    print('argmax:', np.unravel_index(scmap.numpy().argmax(), scmap.shape))

    sample_image_batch = kp_dataset[0][0].unsqueeze(0)
