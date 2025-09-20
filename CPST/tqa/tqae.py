import torch
import sys
from .utils_tqa import _PI_Constructor, adapt_by_error_t, inverse_nonconformity_L1
import numpy as np
from datetime import datetime, timedelta
import os
import util_model


def get_week_matrix(timestamp, matrix_dir='../data/METR/causalgraphs/'):
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


class TQA_E(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_E, self).__init__(base_model_path, **kwargs)
        self.initial_supports = None
        self.base_adj_mx = None
        self.args = None

    def set_supports(self, initial_supports, base_adj_mx, args):
        """Set support variables for dynamic graph processing"""
        self.initial_supports = initial_supports
        self.base_adj_mx = base_adj_mx
        self.args = args

    def predict(self, test_dataset, test_y, scaler, alpha=0.1, state=None, gamma=0.05, update_cal=False, two_sided=True,
                **kwargs):
        if self.initial_supports is None or self.base_adj_mx is None or self.args is None:
            raise ValueError("Please call the 'set supports' method first to set the necessary variables")

        device = test_y.device
        outputs = []

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

                # Use dynamic graph for prediction
                preds = self.base_model(testx, supports).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)  # Predicted y
        yhat = yhat[:test_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)

        B, N, L = predy.shape[0], predy.shape[1], predy.shape[2]

        all_pi, all_predy, all_labelu = [], [], []

        for i in range(N):
            pred = predy[:, i, :].unsqueeze(-1)  # [7067, 12, 1]
            y = test_y[:, i, :].unsqueeze(-1)  # [7067, 12, 1]
            print('{} pred shape {} y shape {}'.format(i, pred.shape, y.shape))

            ret = torch.empty(0).to(device)

            for b in range(y.shape[0]):
                calibration_scores, scores = self.get_nonconformity_scores(self.calibration_preds[:, i, :],
                                                                           self.calibration_truths[:, i, :], pred[b],
                                                                           y[b])
                calibration_scores = calibration_scores.unsqueeze(-1)

                tmp = adapt_by_error_t(pred[b, :, 0], y[b, :, 0], calibration_scores[:, :, 0], alpha=alpha, gamma=gamma,
                                       scores=scores[:, 0], rev_func=inverse_nonconformity_L1,
                                       two_sided=two_sided).T.unsqueeze(-1)
                ret = torch.cat((ret, tmp), dim=0)

            print('ret shape', ret.shape)

            all_pi.append(ret)
            all_predy.append(pred)
            all_labelu.append(y)

        all_pred = torch.cat(all_predy, dim=0)
        all_pi = [torch.tensor(item) for item in all_pi]
        all_ret = torch.cat(all_pi, dim=0)
        all_label = torch.cat(all_labelu, dim=0)

        print('all_pred, all_ret, all_label', all_pred.shape, all_ret.shape, all_label.shape)
        return all_pred, all_ret, all_label
