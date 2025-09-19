import util_model as util
import argparse
from model import *
import numpy as np
import torch
import tqa.tqab as tqa
import os
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='../data/METR-LA/', help='data path')
parser.add_argument('--adjdata', type=str, default='../data/adj_matrices/adj_matrix_la.pkl', help='adj data path')
# parser.add_argument('--adjdata', type=str, default='data/sensor_graph/LA/merged_la.pkl', help='adj data path')
parser.add_argument('--adjtype', type=list,
                    default=['normlap', 'doubletransition', 'doubletransition'],
                    help='list of adj type')
parser.add_argument('--supports_list', type=list, default=[4, 6, 6, 4])

parser.add_argument('--fusion', default='weightsum', help='fusion methods[sum,weightsum,max,mean,average]')
parser.add_argument('--gcn_bool', default=True, action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', default=True, action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', default=True, action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', default='../save/METR/trained_model/best_model.pth', type=str, help='')
parser.add_argument('--save_path', type=str, default='results/results_la_90%_tqba_test.npy', help='save path')
parser.add_argument('--w_1', type=float, default=0.95, help='weight for the first term in TQA')
parser.add_argument('--alpha', type=float, default=0.1, help='significance level for TQA')
parser.add_argument('--update_cal', default=False, action='store_true',
                    help='whether to update calibration during prediction')
args = parser.parse_args()

def main():
    device = torch.device(args.device)

    # Load data
    print("\nLoading data...")
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # Load adjacency matrices
    print("Loading adjacency matrices...")
    adj_matrices, _, _ = util.load_adj_matrix(args.adjdata)
    adj_mx = util.process_adj_matrix(adj_matrices, args.adjtype[0])

    # Create backup matrices
    backup_matrix = np.zeros((args.num_nodes, args.num_nodes))
    np.fill_diagonal(backup_matrix, 1)  # use identity matrix as backup
    base_adj_mx = [adj_mx[0], backup_matrix, backup_matrix]  # use identity matrices as backups

    # Initialize supports
    adj_matrix = torch.tensor(np.array(adj_mx[0])).to(device)
    adj_matrix = torch.unsqueeze(adj_matrix, 0)
    initial_supports = [adj_matrix]

    # Initialize model
    print("\nInitializing model...")
    if args.randomadj:
        adjinit = None
    else:
        adjinit = initial_supports[0]
    if args.aptonly:
        initial_supports = None

    model = CASTMGCN(device=device, num_nodes=args.num_nodes, dropout=args.dropout, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj, aptinit=adjinit, fusion=args.fusion, supports_list=args.supports_list,
                  in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid,
                  skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print("Model loaded successfully")

    # calibration
    val_y = torch.Tensor(dataloader['y_val']).to(device)  # [7066, 12, 617, 2]
    val_y = val_y[:, :, :, 0:1]  # take the first channel

    val_y = val_y.transpose(1, 3)[:, 0, :, :]

    calibration_dataset = dataloader['val_loader']
    cal_model = tqa.TQA_B(model)

    # Set required variables
    cal_model.set_supports(initial_supports, base_adj_mx, args)

    # Directly call calibrate (supports dynamic graph input)
    cal_model.calibrate(calibration_dataset, val_y, scaler, device=device)

    # Prepare test data
    print("\nPreparing test data...")
    test_y = torch.Tensor(dataloader['y_test']).to(device)
    test_y = test_y.transpose(1, 3)[:, 0, :, :]  # ground truth y [7066, 617, 12]
    test_dataset = dataloader['test_loader']

    print("\nStart testing...")
    yhat, pi, label_y = cal_model.predict(test_dataset, test_y, scaler, alpha=args.alpha, w_1=args.w_1,
                                          w_2=1 - args.w_1,
                                          update_cal=args.update_cal)

    # Save results
    data = {
        'yhat': yhat.cpu().detach(),
        'pi': pi.cpu().detach(),
        'label_y': label_y.cpu().detach()
    }
    np.save(args.save_path, data)


if __name__ == '__main__':
    main()
