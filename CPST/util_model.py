import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import h5py


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()

    # avoid division by zero
    d_inv = np.power(rowsum, -1, where=rowsum != 0)
    d_inv[rowsum == 0] = 0.

    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    L = I - D^-1/2 A D^-1/2
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()

    # avoid division by zero
    d_inv_sqrt = np.power(d, -0.5, where=d != 0)
    d_inv_sqrt[d == 0] = 0.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_matrix(pkl_filename):
    """
    Load adjacency matrices from a pickle file
    Args:
        pkl_filename: path to pickle file
    Returns:
        list: a list containing all adjacency matrices
    """
    adj_list = []
    for adj in load_pickle(pkl_filename):
        if not isinstance(adj, np.ndarray):
            adj_mx = adj.numpy()
        else:
            adj_mx = adj
        adj_list.append(adj_mx)
    return adj_list


def process_adj_matrix(adj_mx, adjtype):
    """
    Process adjacency matrix according to the specified type
    Args:
        adj_mx: adjacency matrix
        adjtype: processing type
    Returns:
        list: list of processed adjacency matrices
    """
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adjtype == 'no':
        adj = [np.zeros((207, 207), dtype=np.float32), np.zeros((207, 207), dtype=np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    # convert adj to torch (placeholder)

    return adj


def load_adj(pkl_filename, adjtype):
    """
    Load and process adjacency matrices
    Args:
        pkl_filename: path to pickle file
        adjtype: list of processing types
    Returns:
        list: list of processed adjacency matrices
    """
    adj_list = []
    adj_matrices = load_adj_matrix(pkl_filename)
    for i, adj_mx in enumerate(adj_matrices):
        processed_adj = process_adj_matrix(adj_mx, adjtype[i])
        adj_list.append(processed_adj)
    return adj_list


def load_adj(pkl_filename, adjtype):
    # pre_adj, ac1, ac2 = load_pickle(pkl_filename)  # tensor objects
    # _, _, adjs = load_pickle(pkl_filename)
    adj_list = []
    for i, adj in enumerate(load_pickle(pkl_filename)):

        if not isinstance(adj, np.ndarray):
            adj_mx = adj.numpy()
        else:
            adj_mx = adj

        if adjtype[i] == "scalap":
            adj = [calculate_scaled_laplacian(adj_mx)]
        elif adjtype[i] == "normlap":
            adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
        elif adjtype[i] == "symnadj":
            adj = [sym_adj(adj_mx)]
        elif adjtype[i] == "transition":
            adj = [asym_adj(adj_mx)]
        elif adjtype[i] == "doubletransition":
            adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        elif adjtype[i] == "identity":
            adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
        elif adjtype[i] == 'no':
            adj = [np.zeros((207, 207), dtype=np.float32), np.zeros((207, 207), dtype=np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"
        adj_list.append(adj)

    return adj_list

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '_date.npz'))

        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    # Normalize using train mean and std

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
