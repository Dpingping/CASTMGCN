import torch
import numpy as np

from .utils_tqa import _PI_Constructor


class LASplit(_PI_Constructor):

    def __init__(self, base_model_path=None, alpha=0.1, **kwargs):
        super(LASplit, self).__init__(base_model_path, **kwargs)
        self.alpha = alpha
        self._update_cal_loc = 0


    def calibrate(self, calibration_dataset, val_y, scaler, device=None):
        '''
        Dataset, true labels, scaler
        '''
        self.base_model.eval()
        outputs = []  # Predicted preds

        with torch.no_grad():
            for iter, (x, y) in enumerate(calibration_dataset.get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                preds = self.base_model(testx).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)  # Predicted y - why is the first dimension of predicted shape 7104

        yhat = yhat[:val_y.size(0), ...]


        predy = scaler.inverse_transform(yhat)  # Denormalized prediction results

        self.calibration_preds = torch.nn.Parameter(predy, requires_grad=False)  # Predictions
        self.calibration_truths = torch.nn.Parameter(val_y, requires_grad=False)  # True values

        return
    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        return (cal_pred - cal_y).abs(), (test_pred - test_y).abs()

    def predict(self, test_dataset, test_y, scaler, alpha=0.1, state=None, update_cal=False, **kwargs):

        if alpha is None: alpha = self.alpha

        N = len(self.calibration_preds)
        q = np.ceil((N + 1) * (1 - alpha)) / N
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

        B_1, N_1, L_1 = predy.shape[0], predy.shape[1], predy.shape[2]

        all_pi, all_predy, all_labelu = [], [], []

        for i in range(617):
            predy_new = predy[:, i, :].unsqueeze(-1)  # Predicted data
            test_y_new = test_y[:, i, :].unsqueeze(-1)  # True data

            ret = torch.zeros(B_1, L_1, 2)

            calibration_scores, scores = self.get_nonconformity_scores(
                self.calibration_preds[:, i, :].unsqueeze(-1),
                self.calibration_truths[:, i, :].unsqueeze(-1),
                predy_new, test_y_new)

            width = torch.quantile(calibration_scores, q, dim=0)  # (12,1)

            print('width shape is:', width.shape)
            print('width value is :', width)

            width = width.expand(test_y_new.shape[0], -1, -1)

            print('expand width shape is {}, predy_new shape is {}:'.format(width.shape, predy_new.shape))

            ret[:, :, 0:1] = predy_new - width
            ret[:, :, 1:2] = predy_new + width

            all_pi.append(ret)
            all_predy.append(predy_new)
            all_labelu.append(test_y_new)

        all_pred = torch.cat(all_predy, dim=0)
        all_ret = torch.cat(all_pi, dim=0)
        all_label = torch.cat(all_labelu, dim=0)

        return all_pred, all_ret, all_label
