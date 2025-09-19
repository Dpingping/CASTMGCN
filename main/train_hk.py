# Acknowledgment:
# This code (or parts of it) is adapted from/derived from:
# Graph WaveNet: https://github.com/nnzhan/Graph-WaveNet
# Copyright (c) the authors. Licensed under the MIT License.

import torch
import numpy as np
import argparse
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_prediction')))
import util
from engine import trainer
import os
from datetime import datetime, timedelta
import json

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='mps', help='')
parser.add_argument('--data', type=str, default='../data/HK/', help='data path')
parser.add_argument('--adjdata', type=str, default='../data/adj_matrices/adj_matrix_hk.pkl', help='adj data path')
parser.add_argument('--adjtype', type=list,
                    default=['normlap', 'doubletransition', 'doubletransition'],
                    help='list of adj type')
parser.add_argument('--supports_list', type=list, default=[4, 6, 6, 4])
parser.add_argument('--gcn_bool', default=True, action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', default=False, action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', default=True, action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', default=True, action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--fusion', default='weightsum', help='fusion methods[sum,weightsum,max,mean]')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=64, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=617, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
parser.add_argument('--min_delta', type=float, default=0, help='early stopping min delta')
parser.add_argument('--grad_clip', type=float, default=1, help='gradient clipping value')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--print_every', type=int, default=50, help='print every n iterations')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--save', type=str, default='../save/HK/trained_model/', help='experiment id')

args = parser.parse_args()


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def get_week_matrix(timestamp, matrix_dir='../data/HK/causalgraphs/'):
    try:
        current_date = datetime.fromtimestamp(timestamp)

        # Calculate the week containing the current date
        days_since_monday = current_date.weekday()
        current_monday = current_date - timedelta(days=days_since_monday)
        current_sunday = current_monday + timedelta(days=6)

        # Get the Monday date of the previous week
        last_monday = current_monday - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)

        # Try to load the matrix from the previous week
        last_week_str = last_monday.strftime('%Y%m%d')
        last_week_file = f'week_{last_week_str}.npz'
        last_week_path = os.path.join(matrix_dir, last_week_file)

        if os.path.exists(last_week_path):
            data = np.load(last_week_path, allow_pickle=True)
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
                    thr0 = ratio * np.max(np.abs(matrix1)) if matrix1.size else 0.0
                    thr1 = ratio * np.max(np.abs(matrix2)) if matrix2.size else 0.0
                    matrix1[np.abs(matrix1) < thr0] = 0
                    matrix2[np.abs(matrix2) < thr1] = 0
                    return matrix1, matrix2

        return None, None
    except Exception as e:
        print(f"Error converting timestamp {timestamp}: {e}")
        return None, None


def main():
    # set seed
    torch.manual_seed(2025)
    np.random.seed(2025)
    # load data
    device = torch.device(args.device)  # Use MPS device

    path = args.save
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory:{path}")

    # Load initial adj_mx
    adj_matrices = util.load_adj_matrix('../data/adj_matrices/adj_matrix_hk.pkl')

    adj_mx = util.process_adj_matrix(adj_matrices, 'normlap')
    base_adj_mx = adj_mx.copy()  # Save base matrix as backup

    dataloader = util.load_dataset('../data/HK/', args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # Initialize the supports list, the first one is the fixed adjacency matrix
    adj_matrix = torch.tensor(np.array(adj_mx[0])).to(device)
    adj_matrix = torch.unsqueeze(adj_matrix, 0)
    initial_supports = [adj_matrix]
    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = initial_supports[0]
    if args.aptonly:
        initial_supports = None

    engine = trainer(scaler=scaler, in_dim=args.in_dim, seq_length=args.seq_length, num_nodes=args.num_nodes,
                     nhid=args.nhid, dropout=args.dropout,
                     lrate=args.learning_rate, wdecay=args.weight_decay, device=device, gcn_bool=args.gcn_bool,
                     addaptadj=args.addaptadj,
                     aptinit=adjinit, fusion=args.fusion, supports_list=args.supports_list)

    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_model_state = None
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            batch_timestamps = x[:, 0, 0, 1]
            # Retrieve the corresponding week matrix for each sample
            batch_matrices = []
            for timestamp in batch_timestamps:
                timestamp = int(timestamp)
                matrix1, matrix2 = get_week_matrix(timestamp)
                if matrix1 is not None and matrix2 is not None:
                    # Ensure matrices are 2D
                    if len(matrix1.shape) > 2:
                        matrix1 = matrix1.squeeze()
                    if len(matrix2.shape) > 2:
                        matrix2 = matrix2.squeeze()
                    batch_matrices.append((matrix1, matrix2))
                else:
                    batch_matrices.append((base_adj_mx[1], base_adj_mx[2]))

            # Use the matrix of the first sample (or a more sophisticated strategy)
            matrix1, matrix2 = batch_matrices[0]

            processed_matrix1 = util.process_adj_matrix(matrix1, args.adjtype[1])
            processed_matrix2 = util.process_adj_matrix(matrix2, args.adjtype[2])

            processed_matrix1 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix1]).to(
                device)
            processed_matrix2 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix2]).to(
                device)
            supports = [initial_supports[0], processed_matrix1, processed_matrix2]

            indices = torch.tensor([0, 2])
            trainx = torch.Tensor(x)
            trainx = torch.index_select(trainx, dim=-1, index=indices).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y[..., 0:1]).to(device)
            trainy = trainy.transpose(1, 3)

            supports = [s.to(device) for s in supports]

            metrics = engine.train(trainx, trainy[:, 0, :, :], supports)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
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
                    batch_matrices.append((base_adj_mx[1], base_adj_mx[2]))

            matrix1, matrix2 = batch_matrices[0]
            processed_matrix1 = util.process_adj_matrix(matrix1, args.adjtype[1])
            processed_matrix2 = util.process_adj_matrix(matrix2, args.adjtype[2])

            processed_matrix1 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix1]).to(
                device)
            processed_matrix2 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix2]).to(
                device)
            supports = [initial_supports[0], processed_matrix1, processed_matrix2]
            supports = [s.to(device) for s in supports]

            indices = torch.tensor([0, 2])
            testx = torch.Tensor(x)
            testx = torch.index_select(testx, dim=-1, index=indices).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y[..., 0:1]).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :], supports)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)

        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            best_model_state = engine.model.state_dict()
            torch.save(best_model_state, f"{args.save}epoch_{i}_{mvalid_loss:.2f}.pth")
            print(f"Save best model, valid loss: {mvalid_loss:.4f}")

        early_stopping(mvalid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered, stopping training")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(best_model_state)
    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test'][..., 0:1]).to(device)  # Only take the first channel
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        # Get timestamps and process matrices
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
                batch_matrices.append((base_adj_mx[1], base_adj_mx[2]))

        matrix1, matrix2 = batch_matrices[0]
        processed_matrix1 = util.process_adj_matrix(matrix1, args.adjtype[1])
        processed_matrix2 = util.process_adj_matrix(matrix2, args.adjtype[2])

        processed_matrix1 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix1]).to(device)
        processed_matrix2 = torch.stack([torch.from_numpy(np.array(m)).float() for m in processed_matrix2]).to(device)
        supports = [initial_supports[0], processed_matrix1, processed_matrix2]
        supports = [s.to(device) for s in supports]

        indices = torch.tensor([0, 2])
        testx = torch.Tensor(x)
        testx = torch.index_select(testx, dim=-1, index=indices).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, supports).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # Save prediction results and true values
    pred_list = []
    true_list = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i].cpu().numpy())
        real = realy[:, :, i].cpu().numpy()
        pred = torch.from_numpy(pred).to(device)
        real = torch.from_numpy(real).to(device)
        pred_list.append(pred[..., np.newaxis])
        true_list.append(real[..., np.newaxis])
    pred_all = torch.cat(pred_list, dim=-1)
    true_all = torch.cat(true_list, dim=-1)

    np.savez(f'{args.save}test_pred_and_true.npz',
             pred=pred_all.cpu().numpy(),
             true=true_all.cpu().numpy())
    print(f'Saved the denormalized predicted values and ground truth to  {args.save}test_pred_and_true.npz')

    print("Training finished")
    print("The valid loss on best models is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = pred_all[:, :, i]
        real = true_all[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best models on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(),
               f"{args.save}exp{args.expid}_best_{round(his_loss[bestid], 2)}.pth")

    # Save test results
    test_result = {
        "test_mae": float(np.mean(amae)),
        "test_mape": float(np.mean(amape)),
        "test_rmse": float(np.mean(armse)),
        "mae_list": [float(x) for x in amae],
        "mape_list": [float(x) for x in amape],
        "rmse_list": [float(x) for x in armse]
    }
    with open(f"{args.save}test_results.json", "w") as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)
    print(f"Saved test results to {args.save}test_results.json")

    return np.mean(amae), np.mean(amape), np.mean(armse)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
