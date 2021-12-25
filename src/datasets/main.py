from .mnist import MNIST
#from .cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name, indexes, normal_class, train, data_path, download_data,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                train = train,
                                data_path = data_path,
                                download_data = download_data)


    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path,
                                  normal_class=normal_class,
                                  indexes = indexes,
                                  known_outlier_class=known_outlier_class,
                                  n_known_outlier_classes=n_known_outlier_classes,
                                  ratio_known_normal=ratio_known_normal,
                                  ratio_known_outlier=ratio_known_outlier,
                                  ratio_pollution=ratio_pollution)


    return dataset
