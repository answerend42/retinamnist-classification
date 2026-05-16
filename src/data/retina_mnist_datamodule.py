from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RetinaMNISTDataset(Dataset):
    """Custom dataset for Retina MNIST data loaded from .npz file."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            images: numpy array of shape (N, H, W, C)
            labels: numpy array of shape (N, 1) or (N,)
            transform: optional transform to be applied on a sample
        """
        self.images = images
        self.labels = labels.flatten()  # Ensure labels are 1D
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")

        # Convert numpy array to PIL Image
        image = self.images[idx]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert from HWC to CHW if needed by creating PIL Image
        pil_image = Image.fromarray(image)

        # Apply transforms if any
        if self.transform:
            pil_image = self.transform(pil_image)

        label = int(self.labels[idx])
        return pil_image, label


class RetinaMNISTDataModule(LightningDataModule):
    """`LightningDataModule` for the Retina MNIST dataset.

    The Retina MNIST dataset contains retinal fundus images for disease classification.
    It has 5 classes corresponding to different severity levels of diabetic retinopathy.
    Images are RGB fundus photographs resized to 224x224 pixels.

    A `LightningDataModule` implements 7 key methods:
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        data_file: str = "retinamnist_224.npz",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: int = 224,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize a `RetinaMNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param data_file: The .npz file containing the dataset. Defaults to `"retinamnist_224.npz"`.
        :param batch_size: The batch size. Defaults to `32`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param image_size: The image size. Defaults to `224`.
        :param normalize_mean: Mean values for normalization. Defaults to ImageNet mean.
        :param normalize_std: Standard deviation values for normalization. Defaults to ImageNet std.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of Retina MNIST classes (5).
        """
        return 5

    def prepare_data(self) -> None:
        """Download data if needed.

        This method is called only within a single process on CPU,
        so you can safely add your downloading logic within.
        """
        # Data is already provided in the data directory
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice!

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data_path = f"{self.hparams.data_dir}/{self.hparams.data_file}"

            # Load data from npz file
            data = np.load(data_path)

            # Create datasets
            self.data_train = RetinaMNISTDataset(
                images=data['train_images'],
                labels=data['train_labels'],
                transform=self.train_transform
            )

            self.data_val = RetinaMNISTDataset(
                images=data['val_images'],
                labels=data['val_labels'],
                transform=self.val_test_transform
            )

            self.data_test = RetinaMNISTDataset(
                images=data['test_images'],
                labels=data['test_labels'],
                transform=self.val_test_transform
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = RetinaMNISTDataModule()