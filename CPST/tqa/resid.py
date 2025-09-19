import torch
import numpy as np

from .utils_tqa import _PI_Constructor
import pickle


class LASplit(_PI_Constructor):

    def __init__(self, base_model_path=None, alpha=0.1, **kwargs):
        super(LASplit, self).__init__(base_model_path, **kwargs)
        self.alpha = alpha
        # def __init__(self, base_model_path, alpha=0.1, **kwargs):
        #     super(LASplit, self).__init__()
        #     self.madrnn = torch.load(base_model_path, map_location=kwargs.get('device'))
        #     self.alpha = alpha

        self._update_cal_loc = 0  # if we want to update the calibration residuals in an online fashion

    # def calibrate(self, calibration_dataset: torch.utils.data.Dataset, batch_size=32, device=None):
    #     """
    #     Computes the nonconformity scores for the calibration dataset.
    #     """
    #     self.madrnn.eval()
    #
    #     self.base_model.eval()
    #
    #
    #     calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    #     preds, ys = [], []
    #     with torch.no_grad():
    #         for calibration_example in calibration_loader:
    #             calibration_example = [_.to(device) for _ in calibration_example]
    #             sequences, targets, lengths_input, lengths_target = calibration_dataset._sep_data(calibration_example)
    #             out = self.madrnn(sequences, resid_only=False)
    #             preds.append(out)
    #             ys.append(targets)
    #     self.calibration_preds = torch.nn.Parameter(torch.cat(preds), requires_grad=False)
    #     self.calibration_truths = torch.nn.Parameter(torch.cat(ys), requires_grad=False)
    #
    #     msk = self.calibration_preds[:, 0] > 0
    #     self._min_resid = self.calibration_preds[:, 0][msk].mean()
    #     self.calibration_preds[:, 0] = self.calibration_preds[:, 0].clip(self._min_resid)

    def calibrate(self, calibration_dataset, val_y, scaler, device=None):
        '''
        Dataset, true labels, scaler
        '''
        # Input is already a dataloader

        self.base_model.eval()
        # calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)

        outputs = []  # Predicted preds

        with torch.no_grad():
            for iter, (x, y) in enumerate(calibration_dataset.get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                preds = self.base_model(testx).transpose(1, 3)
                outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)  # Predicted y - why is the first dimension of predicted shape 7104

        yhat = yhat[:val_y.size(0), ...]

        # print('yhat', yhat.shape)

        predy = scaler.inverse_transform(yhat)  # Denormalized prediction results

        self.calibration_preds = torch.nn.Parameter(predy, requires_grad=False)  # Predictions
        self.calibration_truths = torch.nn.Parameter(val_y, requires_grad=False)  # True values

        return

    # def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
    #     print('cal_pred, cal_y, test_pred, test_y', cal_pred.shape, cal_y.shape, test_pred.shape, test_y.shape)
    #     cs = (cal_pred[:, 1] - cal_y).abs() / cal_pred[:, 0]
    #     ts = (test_pred[1] - test_y).abs() / test_pred[0]
    #     return cs, ts
    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        # The most common nonconformity score
        # print(
        #     'cal_pred shape {}, cal_y shape {}, test_pred shape {}, test_y shape {}'.format(cal_pred.shape, cal_y.shape,
        #                                                                                     test_pred.shape,
        #                                                                                     test_y.shape))

        # print('(cal_pred - cal_y).abs(), (test_pred - test_y).abs()', (cal_pred - cal_y).abs().shape,
        #       (test_pred - test_y).abs().shape)
        return (cal_pred - cal_y).abs(), (test_pred - test_y).abs()

    def predict(self, test_dataset, test_y, scaler, alpha=0.1, state=None, update_cal=False, **kwargs):

        if alpha is None: alpha = self.alpha

        N = len(self.calibration_preds)
        print('N', N)

        q = np.ceil((N + 1) * (1 - alpha)) / N
        print('q', q)  # 0.9 quantile
        # resid_yhat = self.madrnn(x, resid_only=False)  # (B, 2, L, 1) predicted y
        #
        # pred = resid_yhat[:, 1]
        # resid_yhat[:, 0] = resid_yhat[:, 0].clip(self._min_resid)
        # assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"

        # ret = torch.clone(resid_yhat)

        device = test_y.device
        outputs = []
        # with torch.no_grad():
        #     for iter, (x, y) in enumerate(test_dataset.get_iterator()):
        #         testx = torch.Tensor(x).to(device)
        #         testx = testx.transpose(1, 3)
        #         preds = self.base_model(testx).transpose(1, 3)
        #         outputs.append(preds.squeeze())
        #
        # yhat = torch.cat(outputs, dim=0)  # predicted y
        # yhat = yhat[:test_y.size(0), ...]
        # predy = scaler.inverse_transform(yhat)

        with torch.no_grad():
            for iter, (x, y) in enumerate(test_dataset.get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                # print('testx value', testx)

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
