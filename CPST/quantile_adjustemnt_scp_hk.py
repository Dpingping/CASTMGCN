import util_model as util
import argparse
from model_hk import *
import numpy as np

from tqa.scp_hk import SCP  # import SCP class directly
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='mps', help='')
parser.add_argument('--data', type=str, default='../data/HK/', help='data path')
parser.add_argument('--adjdata', type=str, default='../data/sensor_graph/adj_matrix_hk.pkl', help='adj data path')
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
parser.add_argument('--save_path', type=str, default='results/CP_comparsion/hk_scp_90%.npy', help='save path')
parser.add_argument('--alpha', type=float, default=0.1, help='significance level for prediction intervals')

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

    # Load model weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    # calibration
    val_y = torch.Tensor(dataloader['y_val']).to(device)
    val_y = val_y.transpose(1, 3)[:, 0, :, :]  # ground truth y [7066, 617, 12]
    calibration_dataset = dataloader['val_loader']

    # Initialize SCP and set dynamic-graph related variables
    cal_model = SCP(model, alpha=args.alpha)
    cal_model.set_supports(initial_supports, base_adj_mx, args)

    # Calibrate
    cal_model.calibrate(calibration_dataset, val_y, scaler, device=device)

    # Prepare test data
    print("\nPreparing test data...")
    test_y = torch.Tensor(dataloader['y_test']).to(device)
    test_y = test_y.transpose(1, 3)[:, 0, :, :]  # ground truth y [7066, 617, 12]
    test_dataset = dataloader['test_loader']

    print("\nStart testing...")
    yhat, pi, label_y = cal_model.predict(test_dataset, test_y, scaler, alpha=args.alpha)

    # Print shapes and sample values for verification
    print("\nValidation data:")
    print("yhat shape:", yhat.shape)
    print("pi shape:", pi.shape)
    print("label_y shape:", label_y.shape)

    # Print first sample for sanity check
    print("\nFirst sample values:")
    print("Prediction:", yhat[0, 0, 0].item())
    print("Ground truth:", label_y[0, 0, 0].item())
    print("Prediction interval:", pi[0, 0, :, 0].tolist())

    # Validate prediction intervals
    print("\nValidate prediction intervals:")
    for i in range(min(5, yhat.shape[0])):  # print first 5 samples
        print(f"\nSample {i}:")
        print(f"Prediction: {yhat[i, 0, 0].item():.2f}")
        print(f"Ground truth: {label_y[i, 0, 0].item():.2f}")
        print(f"PI: [{pi[i, 0, 0, 0].item():.2f}, {pi[i, 0, 1, 0].item():.2f}]")
        print(f"PI width: {pi[i, 0, 1, 0].item() - pi[i, 0, 0, 0].item():.2f}")

    # Save results
    data = {
        'yhat': yhat.cpu().detach().numpy(),  # [nodes_num, batch_size, 12]
        'pi': pi.cpu().detach().numpy(),  # [nodes_num, batch_size, 2, 12]
        'label_y': label_y.cpu().detach().numpy()  # [nodes_num, batch_size, 12]
    }

    # Verify shapes before saving
    print("\nShapes before saving:")
    print("yhat shape:", data['yhat'].shape)
    print("pi shape:", data['pi'].shape)
    print("label_y shape:", data['label_y'].shape)

    # Save results
    np.save(args.save_path, data)
    print(f"\nResults saved to {args.save_path}")

    # Verify saved data
    saved_data = np.load(args.save_path, allow_pickle=True).item()
    print("\nVerify saved data:")
    print("yhat shape:", saved_data['yhat'].shape)
    print("pi shape:", saved_data['pi'].shape)
    print("label_y shape:", saved_data['label_y'].shape)
    print("\nFirst sample values:")
    print("Prediction:", saved_data['yhat'][0, 0, 0])
    print("Ground truth:", saved_data['label_y'][0, 0, 0])
    print("Prediction interval:", saved_data['pi'][0, 0, :, 0].tolist())


if __name__ == '__main__':
    main()
