import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [
        (i, os.path.join(path, (image if isinstance(image, str) else image.decode('UTF-8'))))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
            self,
            num_classes,
            num_samples_per_class,
            batch_type,
            config={},
            device=torch.device("cpu"),
            cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_size: size of meta batch size (e.g. number of functions)
            batch_type: train/val/test
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train: num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val:]
        self.device = device
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / 255.0
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Args:
            does not take any arguments
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and
                2. label batch has shape [K+1, N, N]
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.

            3. The value for `self.num_samples_per_class` will be set to K+1
            since for K-shot classification you need to sample K supports and
            1 query.
        """

        #############################
        #### YOUR CODE GOES HERE ####

        # 1- Sample N different characters from either the specified train, test, or validation folder.
        char_folders = random.sample(self.folders, self.num_classes)

        # 2- Load K + 1 images per character and collect the associated labels, using K images
        # per class for the support set and 1 image per class for the query set.
        labels = np.eye(self.num_classes)
        support_label_path_pairs = get_images(char_folders, labels, nb_samples=self.num_samples_per_class-1, shuffle=False)
        query_label_path_pairs = get_images(char_folders, labels, nb_samples=1, shuffle=True)

        support_images = [self.image_file_to_array(path, -1) for label, path in support_label_path_pairs]
        support_labels = [label for label, path in support_label_path_pairs]
        query_images = [self.image_file_to_array(path, -1) for label, path in query_label_path_pairs]
        query_labels = [label for label, path in query_label_path_pairs]

        # 3- Format the data and return two tensors
        # one of flattened images with shape [K + 1, N, 784]
        images = torch.tensor(np.concatenate((support_images, query_images)))\
            .view(self.num_samples_per_class, self.num_classes, 784)
        # one of one-hot labels [K + 1, N, N ]
        labels = torch.tensor(np.concatenate((support_labels, query_labels)))\
            .view(self.num_samples_per_class, self.num_classes, self.num_classes)

        return (images.float(), labels.float())
        #############################

    def __iter__(self):
        while True:
            yield self._sample()
