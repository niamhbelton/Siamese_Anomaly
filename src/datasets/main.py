from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class,
                                known_outlier_class=known_outlier_class,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path,
                                  normal_class=normal_class,
                                  known_outlier_class=known_outlier_class,
                                  n_known_outlier_classes=n_known_outlier_classes,
                                  ratio_known_normal=ratio_known_normal,
                                  ratio_known_outlier=ratio_known_outlier,
                                  ratio_pollution=ratio_pollution)


    return dataset
