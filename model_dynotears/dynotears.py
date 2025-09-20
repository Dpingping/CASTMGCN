import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm

class DynoTearsMLP(nn.Module):
    def __init__(self, dims, P, bias=True):

        super(DynoTearsMLP, self).__init__()
        assert len(dims) >= 2
        d = dims[0]  # number of variables
        self.P = P  # lag order

        self.dims = dims

        # Allow custom network structure
        assert dims[1] > 0  # ensure hidden layer exists
        self.hidden_layer_size = dims[1]  # set hidden layer size

        # Define learnable parameters

        # use random init
        self.w_est = nn.Parameter(torch.zeros((d, d)))  # W matrix
        self.P_est = nn.ParameterList([nn.Parameter(torch.zeros((d, d))) for _ in range(P)])  # lag matrices

        # Add bias
        self.bias_est = nn.Parameter(torch.zeros(d)) if bias else None

    def forward(self, Xlags):
        """
        :param Xlags: past lagged time series data
        :return: predicted output
        """
        Xlags = Xlags.to(self.w_est.device)

        # Dynamically compute M = X @ W + sum(Xlags @ P)
        M = torch.matmul(Xlags[self.P:], self.w_est)  # current-time W influence

        for i in range(self.P):
            M += torch.matmul(Xlags[self.P - i - 1:-i - 1], self.P_est[i])  # past P-lag terms

        # Add bias
        if self.bias_est is not None:
            M = M + self.bias_est

        return M

    def h_func(self):
        """ DAG constraint """
        d = self.dims[0]
        h = trace_expm(self.w_est * self.w_est) - d
        return h

    def diag_zero(self):
        """ Enforce zero diagonal for W matrix """
        diag_loss = torch.trace(self.w_est * self.w_est)
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



def dual_ascent_step(model, Xlags, device, lambda1, lambda2, lambda3, rho, alpha, h, rho_max, max_opt_steps=5):
    """ Perform one dual ascent step """
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    iteration = 0

    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(Xlags)
            loss = squared_loss(X_hat, Xlags[model.P:], device)

            h_val = model.h_func()
            diag_loss = model.diag_zero()
            penalty1 = 0.5 * rho * h_val * h_val + alpha * h_val

            primal_obj = loss + 100 * penalty1 + 1000 * diag_loss + \
                         lambda1 * L1Norm(model.w_est) + \
                         sum(lambda2 * L1Norm(P) for P in model.P_est)

            # Print current iteration loss periodically
            nonlocal iteration
            if iteration % 10 == 0:  # print every 10 iterations
                print(
                    f"Iteration {iteration}, Loss: {loss.item():.6f}, Primal Objective: {primal_obj.item():.6f}, h_val: {h_val.item():.6f}")
            iteration += 1

            primal_obj.backward()
            return primal_obj

        for _ in range(max_opt_steps):
            optimizer.step(closure)

        with torch.no_grad():
            h_new = model.h_func()
            # Print rho update info
            print(f"Current rho: {rho:.2e}, h_new: {h_new.item():.6f}")

        if h_new.item() > 0.1 * h:
            rho *= 5
            print(f"Increasing rho to: {rho:.2e}")
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new



def dynotears_model(Xlags, P, device, lambda1=0., lambda2=0., lambda3=0., max_iter=10000, h_tol=1e-10, rho_max=1e+18):
    """
    Train DynoTears to learn causal matrices
    :param Xlags: past lagged time series data
    :param P: lag order to estimate
    :return: learned causal matrices
    """
    model = DynoTearsMLP(dims=[Xlags.shape[1], Xlags.shape[0]], P=P).to(device)
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, Xlags, device, lambda1, lambda2, lambda3, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.w_est.detach().cpu().numpy()
    P_est_list = [P.detach().cpu().numpy() for P in model.P_est]
    Z_t = model.bias_est.detach().cpu().numpy()

    return W_est, P_est_list, Z_t


# ===================== Test code =====================
if __name__ == "__main__":
    # Create test data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

    # n, T = 6, 100  # 6 variables, 100 time steps
    np.random.seed(42)
    # Xlags = torch.tensor(np.random.randn(T, n)).float().to(device)

    data = pd.read_csv('data/simulated_data.csv', index_col=False)

    Xlags = torch.tensor(data.to_numpy()).float().to(device)

    print('Xlags shape is', Xlags.shape)

    # Set variable P; e.g., test with P = 2
    P = 2
    W_est, P_est_list = dynotears_model(Xlags, P, device)

    print("Learned W matrix:\n", W_est)
    print('W_est shape', W_est.shape)
    for i, P_mat in enumerate(P_est_list):
        print(f"Learned P{i + 1} matrix:\n", P_mat)
        print(f"Learned P{i + 1} matrix shape is :\n", P_mat.shape)

    # Save W_est and P_est_list to npz file
    np.savez('data/dynotear_predicted.npz', W_est=W_est, P_est_list=P_est_list)
