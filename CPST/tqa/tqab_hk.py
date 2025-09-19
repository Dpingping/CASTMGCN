import torch
import pickle
from .utils_tqa_hk import _PI_Constructor, quantile_regression_EWMA, _torch_rank
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util_model

import os
from datetime import datetime, timedelta


# test

class QuantileRegressionVariants:
    # return a range in [0, 1]
    @classmethod
    def rank_first(cls, pred, y, beta, **kwargs):
        base_scores = (pred - y).abs()
        base_rank = torch.argsort(torch.argsort(base_scores, 1), 1) / (
                base_scores.shape[1] - 1) - 0.5  # get the normalized ranking
        pred_rank = quantile_regression_EWMA(base_rank, beta=beta)
        return pred_rank + 0.5

    @classmethod
    def scale_first(cls, pred, y, beta, **kwargs):
        base_scores = (pred - y).abs()
        pred_aresid = quantile_regression_EWMA(base_scores, beta=beta)
        pred_rank = _torch_rank(pred_aresid, 1) - 0.5
        return pred_rank + 0.5


class BudgetingVariants:
    @classmethod
    def aggressive(cls, test_pred_rank, alpha, max_adj, N):
        assert alpha < 0.5, "The following logic might not make sense for alpha > 0.5 - will need to check later"
        test_pred_rank = test_pred_rank - 0.5  # [0,1] -> [-0.5, 0.5]
        q = (N + 1) * (1 - alpha) / N
        max_adj = (max_adj - q) / alpha
        qb = q + max_adj * alpha * (2 * test_pred_rank)
        return qb

    @classmethod
    def conservative(cls, test_pred_rank, alpha, max_adj, N):
        assert alpha < 0.5, "The following logic might not make sense for alpha > 0.5 - will need to check later"

        q = (N + 1) * (1 - alpha) / N
        max_adj = (max_adj - q) / alpha

        qb = torch.ones_like(test_pred_rank) * q
        delta = test_pred_rank - (1 - alpha)
        adj_up_msk = test_pred_rank > (1 - alpha)
        qb[~adj_up_msk] += ((alpha / (1 - alpha)) ** 2) * delta[~adj_up_msk] * max_adj
        qb[adj_up_msk] += delta[adj_up_msk] * max_adj

        qb[0] = q  # sanify check
        return qb


def get_week_matrix(timestamp, matrix_dir='../data/HK/causalgraphs/'):
    """
    Get corresponding weekly matrix based on timestamp
    """
    try:
        current_date = datetime.fromtimestamp(timestamp)
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
                matrix1, matrix2 = P_est_list[0], P_est_list[1]
                matrix1 = np.squeeze(matrix1)
                matrix2 = np.squeeze(matrix2)
                if len(matrix1.shape) == 2 and len(matrix2.shape) == 2:
                    ratio = 0.3
                    # Calculate threshold (maximum absolute value of each matrix * ratio)
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
            if len(P_est_list) >= 2:
                matrix1, matrix2 = P_est_list[0], P_est_list[1]
                matrix1 = np.squeeze(matrix1)
                matrix2 = np.squeeze(matrix2)
                if len(matrix1.shape) == 2 and len(matrix2.shape) == 2:
                    ratio = 0.3
                    # Calculate threshold (maximum absolute value of each matrix * ratio)
                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        fixed_week_file = 'week_20230703.npz'
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
                    # Calculate threshold (maximum absolute value of each matrix * ratio)
                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        return None, None
    except Exception as e:
        print(f"Error converting timestamp {timestamp}: {e}")
        return None, None


class TQA_B(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_B, self).__init__(base_model_path, **kwargs)
        self.initial_supports = None
        self.base_adj_mx = None
        self.args = None

    def set_supports(self, initial_supports, base_adj_mx, args):
        self.initial_supports = initial_supports
        self.base_adj_mx = base_adj_mx
        self.args = args

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y, beta=0.8):
        raise NotImplementedError()

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha,
                       beta=0.9, max_adj=0.99,
                       **kwargs
                       ):
        device = cal_pred.device
        cal_pred = cal_pred.to(device)
        cal_y = cal_y.to(device)
        test_pred = test_pred.to(device)
        test_y = test_y.to(device)

        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        test_pred_rank = QuantileRegressionVariants.scale_first(pred, y, beta=beta)[:, -1]
        return BudgetingVariants.conservative(test_pred_rank, alpha, max_adj, len(cal_y))

    def predict(self, test_dataset, test_y, scaler, alpha=0.1, w_1=0.95, w_2=0.05, state=None, gamma=0,
                update_cal=True, censor=False, **kwargs):
        if self.initial_supports is None or self.base_adj_mx is None or self.args is None:
            raise ValueError("Please call set_supports method first to set necessary variables")

        device = test_y.device
        outputs = []

        # Prediction process uses torch.no_grad()
        with torch.no_grad():
            for iter, (x, y) in enumerate(test_dataset.get_iterator()):
                # Dynamic adjacency matrix processing
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

                testx = torch.Tensor(x).to(device)
                indices = torch.tensor([0, 2]).to(device)
                testx = torch.index_select(testx, dim=-1, index=indices)
                testx = testx.transpose(1, 3)
                preds = self.base_model(testx, supports).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:test_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)
        B, N, L = predy.shape[0], predy.shape[1], predy.shape[2]

        all_pi, all_predy, all_labelu = [], [], []

        adj_path_la = '../data/adj_matrices/adj_matrix_hk.pkl'
        with open(adj_path_la, 'rb') as f:
            adj_data = pickle.load(f)[0]
            adj_data = torch.tensor(adj_data)
            # print('adj_data shape is ', adj_data.shape)

        # Node processing process uses torch.no_grad()
        with torch.no_grad():
            for i in range(N):  # Process all nodes
                print('{}, predy shape is {},test_y shape is {}'.format(i, predy.shape, test_y.shape))
                # print('np.nonzero(adj_data[i]) is ', np.nonzero(adj_data[i]))
                neighbors = torch.nonzero(adj_data[i], as_tuple=False).squeeze(1).tolist()

                # Modify neighbor node judgment logic
                # if np.nonzero(adj_data[i]).size(0) == 0 and np.nonzero(adj_data[i]).size(1) == 1:
                #     neighbors = []
                # else:
                #     neighbors = np.nonzero(adj_data[i])[0]
                print('neighbors is ', neighbors)

                predy_new = predy[:, i, :].unsqueeze(-1)
                test_y_new = test_y[:, i, :].unsqueeze(-1)
                calibration_preds_point = self.calibration_preds[:, i, :].unsqueeze(-1).to(device)
                calibration_truths_point = self.calibration_truths[:, i, :].unsqueeze(-1).to(device)


                # if len(neighbors) > 0:
                #     neighbor_errors = []
                #     for j in neighbors:
                #         neighbor_pred = self.calibration_preds[:, j, :].unsqueeze(-1).to(device)
                #         neighbor_true = self.calibration_truths[:, j, :].unsqueeze(-1).to(device)
                #         error = (neighbor_pred - neighbor_true).abs()
                #         neighbor_errors.append(error)
                #
                #     neighbor_errors = torch.stack(neighbor_errors, dim=0)
                #     mean_neighbor_errors = neighbor_errors.mean(dim=0)
                #     calibration_scores = (w_1 * (calibration_preds_point - calibration_truths_point).abs() +
                #                           w_2 * mean_neighbor_errors).sort(0)[0]
                # else:
                #     calibration_scores = (calibration_preds_point - calibration_truths_point).abs().sort(0)[0]



                # # Modify neighbor node processing logic to be consistent with reference
                adj_calibration_preds_point = 0
                adj_calibration_truths_point = 0
                for j in neighbors:
                    adj_calibration_preds_point += self.calibration_preds[:, j, :].unsqueeze(-1).to(device)
                    adj_calibration_truths_point += self.calibration_truths[:, j, :].unsqueeze(-1).to(device)

                calibration_scores = (w_1 * (calibration_preds_point - calibration_truths_point) +
                                      w_2 * (adj_calibration_preds_point - adj_calibration_truths_point)).abs().sort(0)[
                    0]


                ret = torch.zeros(B, 2, L, device=device)
                qs = torch.zeros(B, L, device=device)

                if update_cal:
                    for b in range(B):
                        qs[b] = self.get_adjusted_q(calibration_preds_point, calibration_truths_point,
                                                    predy_new[b], test_y_new[b], alpha=alpha, **kwargs)

                        for t in range(L):
                            w = torch.quantile(calibration_scores[:, t, 0], qs[b, t].to(calibration_scores.device))
                            ret[b, 0, t] = predy_new[b, t, 0] - w
                            ret[b, 1, t] = predy_new[b, t, 0] + w

                        calibration_preds_point[self._update_cal_loc] = predy_new[b]
                        calibration_truths_point[self._update_cal_loc] = test_y_new[b]
                        self._update_cal_loc = (self._update_cal_loc + 1) % len(calibration_preds_point)
                else:
                    for b in range(B):
                        qs[b] = self.get_adjusted_q(calibration_preds_point, calibration_truths_point,
                                                    predy_new[b], test_y_new[b], alpha=alpha, **kwargs)

                    for t in range(L):
                        w = torch.quantile(calibration_scores[:, t, 0], qs[:, t].to(calibration_scores.device))
                        ret[:, 0, t] = predy_new[:, t, 0] - w
                        ret[:, 1, t] = predy_new[:, t, 0] + w

                ret = ret.unsqueeze(3)
                all_pi.append(ret)
                all_predy.append(predy_new)
                all_labelu.append(test_y_new)

                # Add code to save intermediate results (optional)
                # data = {
                #     'yhat': predy_new.cpu().detach(),
                #     'pi': ret.cpu().detach(),
                #     'label_y': test_y_new.cpu().detach()
                # }
                # file_path = f'results/LA/tqab_90_{i}.pkl'
                # with open(file_path, 'wb') as f:
                #     pickle.dump(data, f)

        all_pred = torch.cat(all_predy, dim=0)
        all_ret = torch.cat(all_pi, dim=0)
        all_label = torch.cat(all_labelu, dim=0)
        print('all_pred shape {}, all_ret shape {}, all_label shape {}'.format(all_pred.shape, all_ret.shape,
                                                                               all_label.shape))
        return all_pred, all_ret, all_label
