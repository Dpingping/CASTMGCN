import torch
import numpy as np
import util_model
import os
from datetime import datetime, timedelta
from .utils_tqa import _PI_Constructor


def get_week_matrix(timestamp, matrix_dir='../data/METR/causalgraphs/'):
    """
    Get the corresponding weekly matrices based on a UNIX timestamp.
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
                    # Compute thresholds (max absolute value of each matrix * ratio)
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
                    # Compute thresholds (max absolute value of each matrix * ratio)
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
                    # Compute thresholds (max absolute value of each matrix * ratio)
                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        return None, None
    except Exception as e:
        print(f"Error converting timestamp {timestamp}: {e}")
        return None, None


def scores_to_intervals(pred, Y, cal_scores, scales, alpha=0.1):
    # mapping scores back to lower and upper bound.
    assert len(pred.shape) == len(Y.shape) == 1
    N, L = cal_scores.shape
    device = pred.device
    qs = torch.concat([torch.sort(cal_scores, 0)[0], torch.ones([1, L], device=device) * torch.inf], 0)
    qloc = max(0, min(int(np.ceil((1 - alpha) * N)), N))
    w_ts = qs[qloc, :]
    return torch.stack([pred - w_ts * scales, pred + w_ts * scales], 1)


class CFRNN(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(CFRNN, self).__init__(base_model_path, **kwargs)
        self.initial_supports = None
        self.base_adj_mx = None
        self.args = None

    def set_supports(self, initial_supports, base_adj_mx, args):
        self.initial_supports = initial_supports
        self.base_adj_mx = base_adj_mx
        self.args = args

    def predict(self, test_dataset, test_y, scaler, alpha=0.1, state=None, gamma=0.05, update_cal=False, **kwargs):
        if self.initial_supports is None or self.base_adj_mx is None or self.args is None:
            raise ValueError("Please call set_supports() to set required variables before predicting.")

        device = test_y.device
        outputs = []

        with torch.no_grad():
            for iter, (x, y) in enumerate(test_dataset.get_iterator()):
                # Dynamic adjacency matrix handling
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
                processed_matrix1 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix1]).to(device)
                processed_matrix2 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix2]).to(device)
                supports = [self.initial_supports[0], processed_matrix1, processed_matrix2]
                supports = [s.to(device) for s in supports]

                testx = torch.Tensor(x).to(device)
                indices = torch.tensor([0, 2]).to(device)
                testx = torch.index_select(testx, dim=-1, index=indices)
                testx = testx.transpose(1, 3)

                # Predict with dynamic graph
                preds = self.base_model(testx, supports).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)  # predicted y
        yhat = yhat[:test_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)

        B, N, L = predy.shape[0], predy.shape[1], predy.shape[2]

        all_pi, all_predy, all_labelu = [], [], []

        for i in range(N):
            # Calibrate per node
            predy_new = predy[:, i, :].unsqueeze(-1)  # predictions
            test_y_new = test_y[:, i, :].unsqueeze(-1)  # ground truth
            print('node {}, pred shape is {},y shape is {}'.format(i, predy_new.shape, test_y_new.shape))

            ret = []
            for b in range(B):
                calibration_scores, scores = self.get_nonconformity_scores(
                    self.calibration_preds[:, i, :].unsqueeze(-1),
                    self.calibration_truths[:, i, :].unsqueeze(-1),
                    predy_new[b], test_y_new[b])
                res = scores_to_intervals(predy_new[b, :, 0], test_y_new[b, :, 0], calibration_scores[:, :, 0], 1,
                                          alpha=alpha)  # [12,2]
                # res = res.unsqueeze(-1)  # [1,12,2]

                ret.append(res.T.unsqueeze(-1))
                # ret.append(res)  # [7606,12,2]

                if update_cal:
                    self.calibration_preds[self._update_cal_loc] = predy_new[b]
                    self.calibration_truths[self._update_cal_loc] = test_y_new[b]
                    self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)

            ret = torch.stack(ret)

            all_pi.append(ret)

            all_predy.append(predy_new)
            all_labelu.append(test_y_new)

        all_pred = torch.cat(all_predy, dim=0)
        all_ret = torch.cat(all_pi, dim=0)
        all_label = torch.cat(all_labelu, dim=0)

        return all_pred, all_ret, all_label  # , qs.cpu().numpy()
        # return pred, ret
