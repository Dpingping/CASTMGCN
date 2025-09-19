import util_model as util
import argparse
from model_hk import *
import numpy as np

import tqa.tqae_hk as tqa
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='mps', help='')
# parser.add_argument('--data',type=str,default='data/HK/',help='data path')
parser.add_argument('--data', type=str, default='../data/HK/', help='data path')
parser.add_argument('--adjdata', type=str, default='../data/adj_matrices/adj_matrix_hk.pkl', help='adj data path')
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
parser.add_argument('--num_nodes', type=int, default=617, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', default='../save/HK/trained_model/best_model.pth', type=str, help='')
parser.add_argument('--save_path', type=str, default='results/hk_tqae_90%.npy', help='save path')
args = parser.parse_args()


def main():
    device = torch.device(args.device)

    # Load base adjacency matrices
    adj_mx = util.load_adj(args.adjdata, args.adjtype)  # adj_mx is a list
    supports = [torch.tensor(np.array(i)).to(device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # Initialize model with the same parameters as reference
    model = CASTMGCN(device=device,
                  num_nodes=args.num_nodes,
                  dropout=args.dropout,
                  gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj,
                  aptinit=adjinit,
                  fusion=args.fusion,
                  supports_list=args.supports_list,
                  in_dim=args.in_dim,
                  out_dim=args.seq_length,  # use seq_length as output dim
                  residual_channels=args.nhid,  # use nhid as base channels
                  dilation_channels=args.nhid,  # use nhid as base channels
                  skip_channels=args.nhid * 8,  # use 8x nhid as skip channels
                  end_channels=args.nhid * 16)  # use 16x nhid as end channels

    model.to(device)
    # Load model weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('Model loaded successfully')
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # Prepare calibration data
    val_y = torch.Tensor(dataloader['y_val']).to(device)
    val_y = val_y.transpose(1, 3)[:, 0, :, :]  # ground truth y [7066, 617, 12]
    calibration_dataset = dataloader['val_loader']

    # Initialize calibration model and set supports
    cal_model = tqa.TQA_E(model)
    cal_model.set_supports(supports, adj_mx, args)  # set supports for dynamic graph processing
    cal_model.calibrate(calibration_dataset, val_y, scaler, device=device)

    # Prepare test data
    test_y = torch.Tensor(dataloader['y_test']).to(device)
    test_y = test_y.transpose(1, 3)[:, 0, :, :]  # ground truth y
    test_dataset = dataloader['test_loader']

    # Predict
    with torch.no_grad():
        yhat, pi, label_y = cal_model.predict(test_dataset, test_y, scaler, alpha=0.1)

    # Save results
    data = {
        'yhat': yhat.cpu().detach().numpy(),
        'pi': pi.cpu().detach().numpy(),
        'label_y': label_y.cpu().detach().numpy()
    }

    # Ensure save directory exists
    save_dir = args.save_path

    # Save results using np.save to match the reference
    np.save(save_dir, data)


if __name__ == '__main__':
    main()
