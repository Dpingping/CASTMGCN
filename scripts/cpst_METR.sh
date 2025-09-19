python ../CPST/quantile_adjustemnt_tqab_la.py \
  --device cuda:0 \
  --data ../data/METR/ \
  --adjdata ./../data/adj_matrices/adj_matrix_la.pkl \
  --w_1 0.95 \
  --checkpoint ../save/METR/trained_model/<model_name>.pth \
  --save_path results/METR/CPST_0.9_npy /
