import imgaug.augmenters as iaa
import torchvision

import augment
import dataset

LABELLED_IMAGES_FOLDER = '[insert folder name here]'
LABELS_FILE = LABELLED_IMAGES_FOLDER + '[insert path to labels here]'

imgaug_pipeline = iaa.Sequential([
  iaa.Fliplr(0.5), # horizontal flip
  iaa.Flipud(0.5), # vertical flip
  iaa.Sometimes(0.4, iaa.Affine(rotate=(-25, 25))), # 25 degree rotation
  iaa.Sometimes(0.5, iaa.MotionBlur(k = 7)),
  iaa.Sometimes(0.5, iaa.CoarseDropout(0.02, size_percent=0.3, per_channel=0.5)),
  iaa.Sometimes(0.5, iaa.ElasticTransformation(sigma=5)),
  iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        )),
  iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.5, 1.0))),
  iaa.Sometimes(0.1, iaa.AllChannelsHistogramEqualization()),
  iaa.Sometimes(0.1, iaa.AllChannelsCLAHE()),
  iaa.Sometimes(0.1, iaa.Emboss(alpha=[0.0, 1.0], strength=[0.5, 1.5])),
  iaa.Sometimes(0.9, iaa.Crop(px=(0, 16))), # crop images from each side by 0 to 16px (randomly chosen)
  #iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
], random_order=False)

input_transform = torchvision.transforms.Compose([
    augment.ImgaugPipelineTransform(imgaug_pipeline)
])

train_transform = torchvision.transforms.Compose([
  input_transform
])

predict_transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((dataset.KeyPointsDataset._height, dataset.KeyPointsDataset._width)),
])

kp_dataset = dataset.get_key_points_dataset(LABELLED_IMAGES_FOLDER, LABELS_FILE, train_transform)

train_loader, valid_loader = dataset.prepare_train_valid_loaders(kp_dataset, valid_fraction=0.1)