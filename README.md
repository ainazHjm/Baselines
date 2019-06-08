# Baselines
This repo contains the baseline models for image data (hdf5 format) and a simple linear model for sea2sky data (csv format).

To train a new model, use this command in the code's directory with the appropriate arguments:
  `python3 -m main [args]`
If no argument is specified, it will use the default arguments. The following command is an example:
  `python3 -m main --data_path '/home/ainaz/Projects/Landslides/Baselines/code/sea2sky.h5' --threshold 0.9 --model LinearLayer --weight 5 --lr 0.01 --n_epochs 5 --batch_size 50 --num_workers 4 --sea2sky True --feature_num 136 --save_model_to '../models/Linear_threshold90_weighted/' --decay '0.001' --s 1`
To see what each attribute presents, look at the get_args function in main.
To validate a model and plot its ROC and AUC curves, set validate to be true `--validate True`.

To train a model for sea2sky dataset, set the sea2sky flag to true `--sea2sky True`. This flag should remain true for validation as well.
