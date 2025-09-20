python ../CPST/quantile_adjustemnt_tqab_hk.py \
--device cuda:0 \
--data ../data/HK/ \
--adjdata ./../data/adj_matrices/adj_matrix_hk.pkl \
--w_1 0.99 \
--checkpoint ../save/HK/trained_model/<model_name>.pth \
--save_path results/HK/CPST_0.9_npy /

