
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch.utils.data import random_split, ConcatDataset
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric.data.dataset")

warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")

def load_data(dataset_Name):
    if dataset_Name=='proteins':
        data_loaded = TUDataset(root='/tmp/PROTEINS', name='PROTEINS',transform=T.Compose([T.ToUndirected()]))
    elif dataset_Name == 'mutag':
        data_loaded = TUDataset(root='/tmp/MUTAG', name='MUTAG', transform=T.Compose([T.ToUndirected()]))
    elif dataset_Name == 'bzr':
        data_loaded = TUDataset(root='/tmp/BZR', name='BZR', transform=T.Compose([T.ToUndirected()]))
    elif dataset_Name == 'cox2':
        data_loaded = TUDataset(root='/tmp/COX2', name='COX2', transform=T.Compose([T.ToUndirected()]))
    elif dataset_Name == 'imdb-binary':
        data_loaded = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY', transform=T.Compose([T.ToUndirected()]))
    elif dataset_Name == 'imdb-multi':
        data_loaded = TUDataset(root='/tmp/IMDB-MULTI', name='IMDB-MULTI', transform=T.Compose([T.ToUndirected()]))
    elif dataset_Name == 'ptc':
        data_loaded = TUDataset(root='/tmp/PTC_MR', name='PTC_MR', transform=T.Compose([T.ToUndirected()]))
    else:
        raise NotImplementedError
    return data_loaded

# Example usage:
# train_datasets = ["proteins", "mutag", "bzr"]
# test_datasets  = ["cox2", "imdb-binary"]
#
# # Load & merge into one train and one test dataset
# train_data = ConcatDataset([load_data(name) for name in train_datasets])
# test_data  = ConcatDataset([load_data(name) for name in test_datasets])
#
# print(f"Train graphs: {len(train_data)}")
# print(f"Test graphs: {len(test_data)}")