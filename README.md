# Baselines
This repo contains the baseline models for image data in hdf5 format and a simple linear model for sea2sky data in csv format to predict landslides.

## Installation
Assuming that you have python3 and pip3 previously installed, use the following command to install the required packages:

  `pip install -r requirements.txt`
 
## Compile and Run
To train a new model, use this command in the code's directory with the appropriate arguments:

  `python3 -m main [args]`
  
If no argument is specified, it will use the default arguments. The following command is an example:

  `python3 -m main --data_path '/home/ainaz/Projects/Landslides/Baselines/code/sea2sky.h5' --threshold 0.9 --model LinearLayer --weight 5 --lr 0.01 --n_epochs 5 --batch_size 50 --num_workers 4 --sea2sky True --feature_num 136 --save_model_to '../models/Linear_threshold90_weighted/' --decay '0.001' --s 1`
 
Here are the possible arguments with their use cases:
  * `data_path`: Specifies the path to the hdf5 dataset that you have previously created. Use this data format if you have large images.
  * 

To validate a model and plot its ROC and AUC curves, set validate to be true `--validate True`.

To train a model for sea2sky dataset, set the sea2sky flag to true `--sea2sky True`. This flag should remain true for validation as well.

Also, for the sea2sky dataset, there are some helper functions that I wrote for data cleaning and merging multiple csv files (outputs from the matcher) that can be used. The ultimate input data (datapath) to the code is a data table consisting of 136 features, which are available in `sea2sky_features.csv` file.
