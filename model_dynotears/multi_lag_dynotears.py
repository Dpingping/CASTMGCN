# This repository includes code (exact or adapted) from [GraphNOTEARS](https://github.com/googlebaba/GraphNOTEARS).
# We gratefully acknowledge the authors’ work.

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm

class DynoTearsMLP(nn.Module):
    def __init__(self, dims, P, bias=True):
        """
        :param dims: [number of input features d, hidden size]
        :param P: lag order (P=3, 4, 5...)
        """
        super(DynoTearsMLP, self).__init__()
        assert len(dims) >= 2
        d = dims[0]  # number of variables
        self.P = P  # lag order

        self.dims = dims

        # Allow custom network structure
        assert dims[1] > 0  # ensure hidden layer exists
        self.hidden_layer_size = dims[1]  # set hidden layer size
        self.P_est = nn.ParameterList([nn.Parameter(torch.zeros((d, d))) for _ in range(P)])

        # Add bias
        self.bias_est = nn.Parameter(torch.zeros(d)) if bias else None

    def forward(self, Xlags):
        """
        :param Xlags: past lagged time series data
        :return: predicted output
        """
        Xlags = Xlags.to(self.P_est[0].device)

        # Dynamically compute M = sum(Xlags @ P)
        M = torch.zeros_like(Xlags[self.P:])  # initialize M

        for i in range(self.P):
            M += torch.matmul(Xlags[self.P - i - 1:-i - 1], self.P_est[i])  # past P-lag terms

        # Add bias
        if self.bias_est is not None:
            M = M + self.bias_est

        return M

    def h_func(self):
        """ DAG constraint, applied to P_est[0] """
        d = self.dims[0]
        h = trace_expm(self.P_est[0] * self.P_est[0]) - d
        return h

    def diag_zero(self):
        """ Enforce zero diagonal for P_est[0] matrix """
        diag_loss = torch.trace(self.P_est[0] * self.P_est[0])
        return diag_loss


def squared_loss(output, target, device):
    """ Compute squared loss """
    output = output.to(device)
    target = target.to(device)
    n = target.shape[0] * target.shape[1]
    loss = 0.5 / float(n) * torch.sum((output - target) ** 2)
    return loss


def L1Norm(matrix):
    """ L1 regularization """
    return torch.abs(matrix).sum()


def dual_ascent_step(model, Xlags, device, lambda1, lambda2, rho, alpha, h, rho_max, max_opt_steps=5, snapshots=None,
                     last_saved_loss=None, loss_diff_threshold=1e-4, iter_idx=None):
    """ Perform one dual ascent step """
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    iteration = 0
    best_inner_loss = float('inf')
    best_inner_snapshot = None

    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(Xlags)
            loss = squared_loss(X_hat, Xlags[model.P:], device)
            current_loss = loss.item()  # get current loss value

            h_val = model.h_func()
            diag_loss = model.diag_zero()
            penalty1 = 0.5 * rho * h_val * h_val + alpha * h_val

            # Only apply L1 regularization on P_est
            primal_obj = loss + 100 * penalty1 + 1000 * diag_loss + \
                         sum(lambda2 * L1Norm(P) for P in model.P_est)

            # Record best loss and corresponding model state
            nonlocal iteration, best_inner_loss, best_inner_snapshot
            if current_loss < best_inner_loss:
                best_inner_loss = current_loss
                best_inner_snapshot = {
                    'iteration': f"{iter_idx}.{iteration}",
                    'loss': current_loss,
                    'P_est_list': [P.detach().cpu().numpy() for P in model.P_est],
                    'Z_t': model.bias_est.detach().cpu().numpy() if model.bias_est is not None else None,
                    'h': float(h_val),
                    'rho': float(rho),
                    'primal_obj': primal_obj.item()
                }

                # Save snapshot if needed
                if snapshots is not None and last_saved_loss is not None:
                    if abs(current_loss - last_saved_loss) > loss_diff_threshold:
                        snapshots.append(best_inner_snapshot)
                        print(
                            f"Inner iter {iter_idx}.{iteration}: loss = {current_loss:.6f}, loss change = {abs(current_loss - last_saved_loss):.6f}, h = {float(h_val):.6f}, rho = {float(rho):.2e}")

            # Periodically print current iteration loss
            if iteration % 100 == 0:
                print(
                    f"Inner iter {iter_idx}.{iteration}, Loss: {current_loss:.6f}, Primal Objective: {primal_obj.item():.6f}, h_val: {float(h_val):.6f}")
            iteration += 1

            primal_obj.backward()
            return primal_obj

        for _ in range(max_opt_steps):
            optimizer.step(closure)

        with torch.no_grad():
            h_new = model.h_func()
            print(f"Current rho: {rho:.2e}, h_new: {float(h_new):.6f}")

        if h_new.item() > 0.1 * h:
            rho *= 5
            print(f"Increasing rho to: {rho:.2e}")
        else:
            break

    alpha += rho * h_new
    return rho, alpha, h_new, best_inner_snapshot


def dynotears_model(Xlags, P, device, lambda1=0., lambda2=0., max_iter=10000, h_tol=1e-10, rho_max=1e+18,
                    save_interval=100, return_snapshots=False, loss_diff_threshold=1e-4):
    """
    Train DynoTears to learn causal matrices
    :param Xlags: past lagged time series data
    :param P: lag order to estimate
    :param device: compute device
    :param lambda1: L1 regularization coefficient
    :param lambda2: L2 regularization coefficient
    :param max_iter: maximum iterations
    :param h_tol: tolerance for DAG constraint
    :param rho_max: maximum penalty parameter
    :param save_interval: save snapshot every N iterations
    :param return_snapshots: whether to return training snapshots
    :param loss_diff_threshold: threshold to save snapshot when loss difference exceeds this value
    :return: learned causal matrices and optional training snapshots
    """
    model = DynoTearsMLP(dims=[Xlags.shape[1], Xlags.shape[0]], P=P).to(device)
    rho, alpha, h = 1.0, 0.0, np.inf

    # Storage for training snapshots
    snapshots = []
    best_loss = float('inf')
    best_snapshot = None
    last_saved_loss = float('inf')  # last saved loss

    for iter_idx in range(max_iter):
        # Perform dual ascent step
        rho, alpha, h, inner_snapshot = dual_ascent_step(
            model, Xlags, device, lambda1, lambda2,
            rho, alpha, h, rho_max,
            snapshots=snapshots if return_snapshots else None,
            last_saved_loss=last_saved_loss if return_snapshots else None,
            loss_diff_threshold=loss_diff_threshold,
            iter_idx=iter_idx
        )

        # Update last_saved_loss if better snapshot observed
        if inner_snapshot is not None and return_snapshots:
            if inner_snapshot['loss'] < best_loss:
                best_loss = inner_snapshot['loss']
                best_snapshot = inner_snapshot
            last_saved_loss = inner_snapshot['loss']

        if h <= h_tol or rho >= rho_max:
            print(f"Training finished at iteration {iter_idx}: h = {float(h):.6f}, rho = {float(rho):.2e}")
            break

    # Final results
    final_snapshot = {
        'iteration': f"{iter_idx}.final",
        'loss': best_loss,
        'P_est_list': [P.detach().cpu().numpy() for P in model.P_est],
        'Z_t': model.bias_est.detach().cpu().numpy() if model.bias_est is not None else None,
        'h': float(h),
        'rho': float(rho)
    }

    if return_snapshots:
        print("\nTraining statistics:")
        print(f"Total iterations: {iter_idx + 1}")
        print(f"Best loss: {best_loss:.6f} (iteration {best_snapshot['iteration']})")
        print(f"Saved {len(snapshots)} snapshots")
        print(f"Loss difference threshold: {loss_diff_threshold}")

        # Sort snapshots by loss
        snapshots.sort(key=lambda x: x['loss'])
        return final_snapshot['P_est_list'], final_snapshot['Z_t'], snapshots
    else:
        return final_snapshot['P_est_list'], final_snapshot['Z_t']


# ===================== Test code =====================
if __name__ == "__main__":
    # Create test data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

    # n, T = 6, 100  # 6 variables, 100 time steps
    np.random.seed(42)
    # Xlags = torch.tensor(np.random.randn(T, n)).float().to(device)

    data = pd.read_csv('traffic_data_BA_new.csv', index_col=False)

    Xlags = torch.tensor(data.to_numpy()).float().to(device)

    print('Xlags shape is', Xlags.shape)

    # Example lag order
    P = 2
    P_est_list, Z_t = dynotears_model(Xlags, P, device)

    print('P_est_list length:', len(P_est_list))
    for i, P_mat in enumerate(P_est_list):
        print(f"Learned P{i + 1} matrix shape: {P_mat.shape}")

    # Save learned P matrices
    np.savez('data/dynotear_predicted.npz', P_est_list=P_est_list)
