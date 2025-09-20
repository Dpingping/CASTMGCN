import os.path

import torch
import numpy as np, os
from datetime import datetime, timedelta
import util_model

def _torch_rank(x, dim=1):
    return torch.argsort(torch.argsort(x, dim), dim) / (x.shape[dim] - 1)


def inverse_nonconformity_L1(score, pred, t):
    return [pred - score, pred + score]


def quantile_regression_EWMA(_data, beta=0.8):
    w = 1. / torch.pow(beta, torch.arange(len(_data), device=_data.device))
    pred = torch.zeros_like(_data)
    for t in range(1, len(_data)):
        wt = w[-t:]
        wt = wt / wt.sum()
        pred[t] = torch.matmul(wt, _data[:t])
    return pred


def get_week_matrix(timestamp, matrix_dir='../../data/METR/causalgraphs/'):

    try:
        current_date = datetime.fromtimestamp(timestamp)

        # Calculate the week containing the current date
        days_since_monday = current_date.weekday()
        current_monday = current_date - timedelta(days=days_since_monday)
        current_sunday = current_monday + timedelta(days=6)


        last_monday = current_monday - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)


        last_week_str = last_monday.strftime('%Y%m%d')
        last_week_file = f'week_{last_week_str}.npz'
        last_week_path = os.path.join(matrix_dir, last_week_file)

        if os.path.exists(last_week_path):
            data = np.load(last_week_path, allow_pickle=True)
            P_est_list = data['P_est_list']
            if len(P_est_list) >= 2:
                matrix1, matrix2 = P_est_list[0], P_est_list[1]  # Use the first two matrices

                matrix1 = np.squeeze(matrix1)
                matrix2 = np.squeeze(matrix2)
                if len(matrix1.shape) == 2 and len(matrix2.shape) == 2:
                    ratio = 0.3

                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        current_week_str = current_monday.strftime('%Y%m%d')
        current_week_file = f'week_{current_week_str}.npz'
        current_week_path = os.path.join(matrix_dir, current_week_file)

        if os.path.exists(current_week_path):
            data = np.load(current_week_path, allow_pickle=True)
            P_est_list = data['P_est_list']
            # Check the length of P_est_list
            if len(P_est_list) >= 2:
                matrix1, matrix2 = P_est_list[0], P_est_list[1]  # Use the first two matrices
                # Ensure matrices are 2D
                matrix1 = np.squeeze(matrix1)
                matrix2 = np.squeeze(matrix2)
                if len(matrix1.shape) == 2 and len(matrix2.shape) == 2:
                    ratio = 0.3

                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        fixed_week_file = 'week_20120305.npz'
        fixed_week_path = os.path.join(matrix_dir, fixed_week_file)
        if os.path.exists(fixed_week_path):
            data = np.load(fixed_week_path, allow_pickle=True)
            P_est_list = data['P_est_list']
            if len(P_est_list) >= 2:
                matrix1, matrix2 = P_est_list[0], P_est_list[1]
                matrix1 = np.squeeze(matrix1)
                matrix2 = np.squeeze(matrix2)
                if len(matrix1.shape) == 2 and len(matrix2.shape) == 2:
                    ratio = 0.3

                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        return None, None
    except Exception as e:
        print(f"Error converting timestamp {timestamp}: {e}")
        return None, None


class _PI_Constructor(torch.nn.Module):
    def __init__(self, base_model=None, **kwargs):
        super(_PI_Constructor, self).__init__()
        self.base_model = base_model


        for param in self.base_model.parameters(): param.requires_grad = False

        self.kwargs = kwargs

        self._update_cal_loc = 0  # if we want to update the calibration residuals in an online fashion

    def fit(self):
        raise NotImplementedError()

    def calibrate(self, calibration_dataset, val_y, scaler, device=None):

        self.base_model.eval()
        outputs = []

        with torch.no_grad():
            for iter, (x, y) in enumerate(calibration_dataset.get_iterator()):

                batch_timestamps = x[:, 0, 0, 1]
                batch_matrices = []
                for timestamp in batch_timestamps:
                    timestamp = int(timestamp)
                    matrix1, matrix2 = get_week_matrix(timestamp)
                    if matrix1 is not None and matrix2 is not None:
                        if len(matrix1.shape) > 2:
                            matrix1 = matrix1.squeeze()
                        if len(matrix2.shape) > 2:
                            matrix2 = matrix2.squeeze()
                        batch_matrices.append((matrix1, matrix2))
                    else:
                        batch_matrices.append((self.base_adj_mx[1], self.base_adj_mx[2]))

                matrix1, matrix2 = batch_matrices[0]
                processed_matrix1 = util_model.process_adj_matrix(matrix1, self.args.adjtype[1])
                processed_matrix2 = util_model.process_adj_matrix(matrix2, self.args.adjtype[2])
                processed_matrix1 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix1]).to(
                    device)
                processed_matrix2 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix2]).to(
                    device)
                supports = [self.initial_supports[0], processed_matrix1, processed_matrix2]
                supports = [s.to(device) for s in supports]

                indices = torch.tensor([0, 2])
                testx = torch.Tensor(x)
                testx = torch.index_select(testx, dim=-1, index=indices).to(device)
                testx = testx.transpose(1, 3)
                preds = self.base_model(testx, supports).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:val_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)

        self.calibration_preds = torch.nn.Parameter(predy, requires_grad=False)
        self.calibration_truths = torch.nn.Parameter(val_y, requires_grad=False)
        return

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        # The most common nonconformity score
        return (cal_pred - cal_y).abs(), (test_pred - test_y).abs()

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha, **kwargs):
        raise NotImplementedError()

    def predict(self, x, y, alpha=0.05, **kwargs):
        raise NotImplementedError()


def mask_scores(scores, scores_len):
    for i in range(len(scores_len)):
        scores[i, scores_len[i]:] = 0
    return scores


def adapt_by_error_t(pred, Y, cal_scores, cal_scores_len=None, gamma=0.005, alpha=0.1,
                     *,
                     scores=None,
                     rev_func=None,
                     two_sided=True,
                     # mapping scores back to lower and upper bound. Take argument (score, pred, t)
                     ):
    if scores is None and rev_func is None:
        scores = (pred - Y).abs()
        rev_func = inverse_nonconformity_L1
    else:
        assert not (scores is None or rev_func is None)
    assert len(pred.shape) == len(Y.shape) == 1
    L = len(Y)
    device = pred.device
    if cal_scores_len is not None:
        cal_scores = mask_scores(cal_scores, cal_scores_len)
        _sidx = torch.argsort(cal_scores_len, descending=True)
        cal_scores, cal_scores_len = cal_scores[_sidx], cal_scores_len[_sidx]

        ns = []
        qs = []
        for t in range(L):
            n = (cal_scores_len > t).int().sum().item()
            qs.append(torch.concat(
                [torch.sort(cal_scores[:n, t], descending=False)[0], torch.ones(1, device=device) * torch.inf]))
            ns.append(n)
    else:
        ns = [cal_scores.shape[0]] * L
        qs = []
        for t in range(L):
            qs.append(torch.concat(
                [torch.sort(cal_scores[:, t], descending=False)[0], torch.ones(1, device=device) * torch.inf]))

    def Q(a, t):
        q = 1 - a
        vs = qs[t]
        n = ns[t]
        loc = torch.ceil(q * (n)).long().clip(0, len(vs) - 1)
        return vs[loc]

    a_ts = torch.ones(L + 1, device=device) * alpha
    err_ts = torch.empty(L, dtype=torch.float, device=device)
    w_ts = torch.empty(L, dtype=torch.float, device=device)

    pred_pis = []
    for t in range(L):
        w_ts[t] = Q(a_ts[t], t)  # Get the current adjusted nonconformity score
        if gamma > 0:
            s_t = scores[t]  # get the actual nonconformity score
            err_ts[t] = (s_t > w_ts[t]).int()  # check if it's violated
            if (two_sided and (a_ts[t] > 1 or a_ts[t] < 0)) or ((not two_sided) and (a_ts[t] > 1)):
                a_ts[t + 1] = a_ts[t] + gamma * (alpha - a_ts[t])  # a_{t+1}
            else:
                a_ts[t + 1] = a_ts[t] + gamma * (alpha - err_ts[t])  # a_{t+1}
        pred_pis.append(rev_func(w_ts[t], pred[t], t))
    return torch.tensor(pred_pis, device=device)
