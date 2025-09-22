import torch
import pickle
from .utils import _PI_Constructor, quantile_regression_EWMA, _torch_rank
import numpy as np


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


class TQA_B(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_B, self).__init__(base_model_path, **kwargs)

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y, beta=0.8):
        raise NotImplementedError()

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha,
                       beta=0.9, max_adj=0.99,
                       **kwargs
                       ):

        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T

        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        # Using the scale-based prediction for rank
        test_pred_rank = QuantileRegressionVariants.scale_first(pred, y, beta=beta)[:,
                         -1]
        # Use the conservative adjustment

        return BudgetingVariants.conservative(test_pred_rank, alpha, max_adj, len(cal_y))  # Return adjusted quantile

    def predict(self, test_dataset, test_y, scaler, alpha=0.1, w_1=0.95, w_2=0.05, state=None, gamma=0,
                update_cal=True,
                censor=False,
                **kwargs):
        # assert  len(x.shape) == 3, "(batch, length, features)"
        device = test_y.device

        outputs = []
        with torch.no_grad():
            for iter, (x, y) in enumerate(test_dataset.get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)

                preds = self.base_model(testx).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)  # Predicted y
        yhat = yhat[:test_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)

        B, N, L = predy.shape[0], predy.shape[1], predy.shape[2]

        all_pi, all_predy, all_labelu = [], [], []

        adj_path = 'data/sensor_graph/merged_file.pkl'
        with open(adj_path, 'rb') as f:
            adj_data = pickle.load(f)[0]
            print('adj_data shape is ', adj_data.shape)
        # Prediction values are already available
        for i in range(N):  # Node
            # Find neighbors
            print('{}, predy shape is {},test_y shape is {}'.format(i, predy.shape, test_y.shape))
            print('np.nonzero(adj_data[i]) is ', np.nonzero(adj_data[i]))
            if np.nonzero(adj_data[i]).size(0) == 0 and np.nonzero(adj_data[i]).size(1) == 1:
                neighbors = []
                print('neighbors is ', neighbors)
            else:
                neighbors = np.nonzero(adj_data[i])[0]
            print('neighbors is ', neighbors)

            predy_new = predy[:, i, :].unsqueeze(-1)
            test_y_new = test_y[:, i, :].unsqueeze(-1)

            calibration_preds_point = self.calibration_preds[:, i, :].unsqueeze(
                -1)
            calibration_truths_point = self.calibration_truths[:, i, :].unsqueeze(
                -1)

            adj_calibration_preds_point = 0
            adj_calibration_truths_point = 0

            for j in neighbors:
                adj_calibration_preds_point += self.calibration_preds[:, j, :].unsqueeze(-1)
                adj_calibration_truths_point += self.calibration_truths[:, j, :].unsqueeze(-1)

            calibration_scores = (w_1 * (calibration_preds_point - calibration_truths_point) + w_2 * (
                    adj_calibration_preds_point - adj_calibration_truths_point)).abs().sort(0)[
                0]  # calibration_scores calculated all calibration scores shape [7066, 12, 1] calibration

            ret = torch.zeros(B, 2, L, device=device)  # 7067 2 12
            qs = torch.zeros(B, L, device=device)  # 7067 12

            if update_cal:  #
                for b in range(B):  # Calculate once for each data sample
                    # get the adjusted quantile - get adjusted quantile, adjust separately for each sample
                    qs[b] = self.get_adjusted_q(calibration_preds_point, calibration_truths_point, predy_new[b],
                                                test_y_new[b],
                                                alpha=alpha,
                                                **kwargs)  # Adjusted quantile

                    for t in range(L):  # 12 time steps
                        w = torch.quantile(calibration_scores[:, t, 0], qs[
                            b, t])  # calibration_scores[:, t, 0] calibration score for all samples at time step t, qs[b, t] adjusted quantile, resulting quantile_value
                        ret[b, 0, t] = predy_new[b, t, 0] - w
                        ret[b, 1, t] = predy_new[b, t, 0] + w

                    calibration_preds_point[self._update_cal_loc] = predy_new[b]  # Update residuals
                    calibration_truths_point[self._update_cal_loc] = test_y_new[b]  # Update residuals
                    self._update_cal_loc = (self._update_cal_loc + 1) % len(calibration_preds_point)
            else:
                qs = torch.zeros(B, L, device=device) # 7067 12
                for b in range(B):
                    qs[b] = self.get_adjusted_q(calibration_preds_point, calibration_truths_point, predy_new[b],
                                                test_y_new[b],
                                                alpha=alpha,
                                                **kwargs) # Adjusted quantile - how to adjust quantile?

                for t in range(L): # 12 time steps
                    w = torch.quantile(calibration_scores[:, t, 0], qs[:, t]) # calibration_scores[:, t, 0] calibration score for all samples at time step t, qs[b, t] adjusted quantile, resulting quantile_value
                    ret[:, 0, t] = predy_new[:, t, 0] - w
                    ret[:, 1, t] = predy_new[:, t, 0] + w

            ret = ret.unsqueeze(3)
            all_pi.append(ret)
            all_predy.append(predy_new)
            all_labelu.append(test_y_new)

            data = {
                'yhat': predy_new.cpu().detach(),
                'pi': ret.cpu().detach(),
                'label_y': test_y_new.cpu().detach()
            }


        all_pred = torch.cat(all_predy, dim=0)
        all_ret = torch.cat(all_pi, dim=0)
        all_label = torch.cat(all_labelu, dim=0)
        print('all_pred shape {}, all_ret shape {}, all_label shape {}'.format(all_pred.shape, all_ret.shape,
                                                                               all_label.shape))
        return all_pred, all_ret, all_label
