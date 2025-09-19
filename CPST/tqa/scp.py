import torch
import numpy as np
import os
from datetime import datetime, timedelta
import util_model

from .utils_tqa import _PI_Constructor
import pickle


def get_week_matrix(timestamp, matrix_dir='../data/METR/causalgraphs/'):
    """Get corresponding weekly matrix based on timestamp
    
    Args:
        timestamp: timestamp
        matrix_dir: matrix file directory
        
    Returns:
        matrix1, matrix2: two adjacency matrices
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


class SCP(_PI_Constructor):
    """Split Conformal Prediction implementation
    
    Although SCP itself is naive conformal prediction, the base model (gwnet) requires dynamic graph input.
    Therefore, dynamic graph support functionality is retained.
    """

    def __init__(self, base_model, alpha=0.1, **kwargs):
        """Initialize SCP
        
        Args:
            base_model: base model
            alpha: significance level, default is 0.1
            **kwargs: other parameters
        """
        super(SCP, self).__init__(base_model, **kwargs)
        self.alpha = alpha
        self.initial_supports = None
        self.base_adj_mx = None
        self.args = None
        self._update_cal_loc = 0  # Used for online calibration residual updates

    def set_supports(self, initial_supports, base_adj_mx, args):
        """Set variables required for dynamic graphs
        
        Args:
            initial_supports: initial supports list
            base_adj_mx: base adjacency matrix
            args: parameter object
        """
        self.initial_supports = initial_supports
        self.base_adj_mx = base_adj_mx
        self.args = args

    def calibrate(self, calibration_dataset, val_y, scaler, device=None):
        """Calibration process
        
        Args:
            calibration_dataset: calibration dataset
            val_y: validation set true values
            scaler: data scaler
            device: computing device
        """
        self.base_model.eval()
        outputs = []

        with torch.no_grad():
            for iter, (x, y) in enumerate(calibration_dataset.get_iterator()):
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

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:val_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)

        self.calibration_preds = torch.nn.Parameter(predy, requires_grad=False)
        self.calibration_truths = torch.nn.Parameter(val_y, requires_grad=False)

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        """Calculate nonconformity scores
        
        Use absolute error between predicted and true values as nonconformity score
        
        Args:
            cal_pred: calibration set predicted values
            cal_y: calibration set true values
            test_pred: test set predicted values
            test_y: test set true values
            
        Returns:
            calibration_scores: calibration set nonconformity scores
            test_scores: test set nonconformity scores
        """
        return (cal_pred - cal_y).abs(), (test_pred - test_y).abs()

    def predict(self, test_dataset, test_y, scaler, alpha=None, state=None, update_cal=False, **kwargs):
        """Prediction process
        
        Perform naive CP algorithm separately for each node, using calibration set to determine width for each node.
        For each node, calculate nonconformity scores for that node across all calibration samples.
        
        Args:
            test_dataset: test dataset
            test_y: test set true values
            scaler: data scaler
            alpha: significance level, if None use initialization value
            state: state (unused)
            update_cal: whether to update calibration set (unused)
            **kwargs: other parameters
            
        Returns:
            all_pred: predicted values, shape [nodes_num, batch_size, 12]
            all_ret: prediction intervals, shape [nodes_num, batch_size, 2, 12]
            all_label: true values, shape [nodes_num, batch_size, 12]
        """
        if alpha is None:
            alpha = self.alpha
            
        # Check if dynamic graph required variables are set
        if self.initial_supports is None or self.base_adj_mx is None or self.args is None:
            raise ValueError("Please call set_supports method first to set variables required for dynamic graphs")

        device = test_y.device
        outputs = []
        
        # Step 1: Use dynamic graph for prediction
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

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:test_y.size(0), ...]
        predy = scaler.inverse_transform(yhat)

        B_1, N_1, L_1 = predy.shape[0], predy.shape[1], predy.shape[2]
        print('Prediction result shape:', predy.shape)  # [6850, 207, 12]

        # Step 2: Perform CP algorithm separately for each node
        all_pi = torch.zeros(N_1, B_1, 2, L_1, device=device)  # [nodes_num, batch_size, 2, 12]
        all_predy = torch.zeros(N_1, B_1, L_1, device=device)  # [nodes_num, batch_size, 12]
        all_labelu = torch.zeros(N_1, B_1, L_1, device=device)  # [nodes_num, batch_size, 12]
        
        # Calculate prediction intervals for each node
        for i in range(N_1):
            print(f'\nProcessing node {i}')
            # Get predicted and true values for current node
            predy_new = predy[:, i, :]  # [B_1, 12]
            test_y_new = test_y[:, i, :]  # [B_1, 12]
            
            # Get predicted and true values for current node on calibration set
            cal_pred_i = self.calibration_preds[:, i, :]  # [N_cal, 12]
            cal_true_i = self.calibration_truths[:, i, :]  # [N_cal, 12]
            
            # Validate calibration set data
            print(f'Calibration set predicted value range: [{cal_pred_i.min().item():.2f}, {cal_pred_i.max().item():.2f}]')
            print(f'Calibration set true value range: [{cal_true_i.min().item():.2f}, {cal_true_i.max().item():.2f}]')
            
            # Calculate nonconformity scores for current node on calibration set
            calibration_scores = (cal_pred_i - cal_true_i).abs()  # [N_cal, 12]
            print('Calibration score shape:', calibration_scores.shape)
            print(f'Calibration score range: [{calibration_scores.min().item():.2f}, {calibration_scores.max().item():.2f}]')
            
            # Calculate quantile
            n_cal = calibration_scores.size(0)
            q = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            print(f'Quantile q: {q:.4f}')
            
            # Calculate quantile separately for each time step
            widths = []
            for t in range(L_1):
                # Get calibration scores for current time step
                scores_t = calibration_scores[:, t]  # [N_cal]
                # Validate scores
                print(f'Time step {t} calibration score range: [{scores_t.min().item():.2f}, {scores_t.max().item():.2f}]')
                # Calculate quantile
                width_t = torch.quantile(scores_t, q)  # scalar
                print(f'Time step {t} width: {width_t.item():.2f}')
                widths.append(width_t)
            width = torch.tensor(widths, device=device)  # [12]
            
            # Validate prediction intervals
            print('\nValidating prediction intervals:')
            for t in range(L_1):
                # Ensure width[t] is scalar
                width_t = width[t].item()
                # Calculate prediction intervals
                all_pi[i, :, 0, t] = predy_new[:, t] - width_t  # lower bound
                all_pi[i, :, 1, t] = predy_new[:, t] + width_t  # upper bound
                
                # Validate prediction interval for first sample
                if t == 0:
                    print(f'First sample predicted value at time step {t}: {predy_new[0, t].item():.2f}')
                    print(f'First sample true value at time step {t}: {test_y_new[0, t].item():.2f}')
                    print(f'First sample prediction interval at time step {t}: [{all_pi[i, 0, 0, t].item():.2f}, {all_pi[i, 0, 1, t].item():.2f}]')
                    print(f'First sample interval width at time step {t}: {width_t:.2f}')
            
            # Store predicted and true values
            all_predy[i] = predy_new
            all_labelu[i] = test_y_new
            
            # Calculate coverage
            coverage = ((test_y_new >= all_pi[i, :, 0, :]) & (test_y_new <= all_pi[i, :, 1, :])).float().mean()
            print(f'Node {i} coverage: {coverage.item():.4f}')

        # Calculate overall coverage
        total_coverage = ((all_labelu >= all_pi[:, :, 0, :]) & (all_labelu <= all_pi[:, :, 1, :])).float().mean()
        print('\nOverall coverage:', total_coverage.item())

        return all_predy, all_pi, all_labelu
