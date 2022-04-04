from .mnist import MNIST
from .cifar10 import CIFAR10
from .mnist_fashion import FASHION


def load_dataset(dataset_name, indexes, normal_class, task, data_path, download_data,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'fashion')
    assert dataset_name in implemented_datasets

    dataset = None


    if dataset_name == 'mnist':
        dataset = MNIST(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                download_data = download_data)





    if dataset_name == 'cifar10':
        dataset = CIFAR10(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                download_data = download_data)

    if dataset_name == 'fashion':
        dataset = FASHION(indexes = indexes,
                                root=data_path,
                                normal_class=normal_class,
                                task = task,
                                data_path = data_path,
                                download_data = download_data)


    return dataset
