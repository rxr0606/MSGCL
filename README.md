# MSGCL
MSGCL
please run by：#nohup python -u main.py -dataset MDAD -k_fold 5 -sparse 0 -epochs_cls 600 -lr_cls 0.001 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.25 -nlayers_cls 2 -patience_cls 10 -epochs 4000 -lr 0.01 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner fgp -k 30 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 100 -tau 1 -maskfeat_rate_learner 0.5 -maskfeat_rate_anchor 0.7 -contrast_batch_size 0 -c 0 
