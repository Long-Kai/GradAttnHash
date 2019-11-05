import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def get_transform(ds_name):
    if ds_name == "cifar":
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # normalize for AlexNet
        transformer = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          normalize])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # normalize for AlexNet
        transformer = transforms.Compose([ResizeImage(256),  # transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    return transformer


def get_test_transform(ds_name, test_10_crop=False):
    if ds_name == "cifar":
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # normalize for AlexNet
        transformer = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          normalize])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # normalize for AlexNet
        if test_10_crop:
            resize_size = 256
            crop_size = 224
            start_first = 0
            start_center = (resize_size - crop_size - 1) / 2
            start_last = resize_size - crop_size - 1
            transformer = {}
            transformer['val0'] = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(p=1.0),  # force flip
                PlaceCrop(crop_size, start_first, start_first),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val1'] = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(p=1.0),  # force flip
                PlaceCrop(crop_size, start_last, start_last),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val2'] = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(p=1.0),  # force flip
                PlaceCrop(crop_size, start_last, start_first),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val3'] = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(p=1.0),  # force flip
                PlaceCrop(crop_size, start_first, start_last),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val4'] = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(p=1.0),  # force flip
                PlaceCrop(crop_size, start_center, start_center),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val5'] = transforms.Compose([
                transforms.Resize(resize_size),
                PlaceCrop(crop_size, start_first, start_first),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val6'] = transforms.Compose([
                transforms.Resize(resize_size),
                PlaceCrop(crop_size, start_last, start_last),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val7'] = transforms.Compose([
                transforms.Resize(resize_size),
                PlaceCrop(crop_size, start_last, start_first),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val8'] = transforms.Compose([
                transforms.Resize(resize_size),
                PlaceCrop(crop_size, start_first, start_last),
                transforms.ToTensor(),
                normalize
            ])
            transformer['val9'] = transforms.Compose([
                transforms.Resize(resize_size),
                PlaceCrop(crop_size, start_center, start_center),
                transforms.ToTensor(),
                normalize
            ])
        else:
            start_center = (256 - 224 - 1) / 2
            transformer = transforms.Compose([transforms.Resize(256),
                                              PlaceCrop(224, start_center, start_center),
                                              # transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])

    return transformer


def get_train_data(ds_config, seed=179):
    torch.manual_seed(seed)  # for testing performance
    dset = ImageList(open(ds_config["list_path"]).readlines(), transform=get_transform(ds_config["dataset"]))
    ds_loader = torch.utils.data.DataLoader(dset, batch_size=ds_config["batch_size"],
                                            shuffle=True, num_workers=4)
    return ds_loader


def get_test_query(ds_config, test_10_crop=False, seed=2251):
    torch.manual_seed(seed)  # for testing performance
    if test_10_crop:
        transformer = get_test_transform(ds_config["dataset"], test_10_crop)
        ds_loader = {}
        for i in range(10):
            dset = ImageList(open(ds_config["list_path"]).readlines(), transform=transformer["val" + str(i)])
            ds_loader["val" + str(i)] = torch.utils.data.DataLoader(dset, batch_size=ds_config["batch_size"],
                                                                    shuffle=False, num_workers=4)

    else:
        dset = ImageList(open(ds_config["list_path"]).readlines(), transform=get_test_transform(ds_config["dataset"]))
        ds_loader = torch.utils.data.DataLoader(dset, batch_size=ds_config["batch_size"],
                                                shuffle=True, num_workers=4)
    return ds_loader


def get_test_db(ds_config, test_10_crop=False, seed=4099):
    torch.manual_seed(seed)
    if test_10_crop:
        transformer = get_test_transform(ds_config["dataset"], test_10_crop)

        ds_loader = {}
        for i in range(10):
            dset = ImageList(open(ds_config["list_path"]).readlines(), transform=transformer["val" + str(i)])
            ds_loader["val" + str(i)] = torch.utils.data.DataLoader(dset, batch_size=ds_config["batch_size"],
                                                                    shuffle=False, num_workers=4)

    else:
        dset = ImageList(open(ds_config["list_path"]).readlines(), transform=get_test_transform(ds_config["dataset"]))
        ds_loader = torch.utils.data.DataLoader(dset, batch_size=ds_config["batch_size"],
                                                shuffle=True, num_workers=4)
    return ds_loader


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
