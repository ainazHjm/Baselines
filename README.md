# Baselines
This repo contains the baseline models for image data in hdf5 format and a simple linear model for sea2sky data in csv format to predict landslides.

## Installation
Assuming that you have python3 and pip3 previously installed, use the following command to install the required packages:

  `pip install -r requirements.txt`
 
## Compile and Run
Clone the `sea2sky` branch into a folder. This folder should have access to the data folder so don't put it in a write protected place. Get the corresponding data containing the csv files and previously created dataset (.h5) and put them in a separate folder as the code.

To train a new model, use this command in the code's directory with the appropriate arguments:

  `python3 -m main [args]`
  
If no argument is specified, it will use the default arguments. The following command is an example:

  `python3 -m main --data_path '/home/ainaz/Projects/Landslides/Baselines/code/sea2sky.h5' --threshold 0.9 --model LinearLayer --weight 5 --lr 0.01 --n_epochs 5 --batch_size 50 --num_workers 4 --sea2sky True --feature_num 136 --save_model_to '../models/Linear_threshold90_weighted/' --decay '0.001' --s 1`
 
Here are the possible arguments with their use cases:
  * `pos_weight`: Specifies the positive weight of the loss function (e.g. BCEwithLogitsLoss) to balance the dataset.
  * `threshold`: The threshold to assign binary class labels to the probabilities from sea2sky dataset.
  * `decay`: The regularization parameter to avoid overfitting.
  * `model`: The name of the model to be used for training.
  * `lr`: The learning rate of the optimizer (neither too small nor too big).
  * `n_epochs`: The number of epochs to be used to train the model.
  * `batch_size`: The number of samples to be fed to the network at the same time (pick a larger batch size for noisy loss).
  * `num_workers`: The number of workers for faster data loading with pytorch DataLoader (usually half the batch size).
  * `load_model`: The path to the trained model to get the predictions.
  * `data_path`: Specifies the path to the hdf5 dataset that you have previously created. Use this data format if you have large images.
  * `region`: The name of the region from where the dataset is produced (not used for sea2sky).
  * `pix_res`: The pixel resolution of the elevation map (DEM) which is not used for sea2sky.
  * `save`: Specifies how often the model should be saved (save being 10 means saving the trained model every 10 epochs).
  * `feature_num`: The total number of features/inputs.
  * `save_res_to`: Specifies the path to save the predictions/results to.
  * `validate`: Set to `True` to validate a trained model and get the predictions. You can also write the predictions into a csv file and plot the ROC and AUC curves by calling `write_results(...)` and `plot_curves(...)` on the outputs of the validate function.
  * `path2json`: For preprocessing and creating the hdf5 dataset.

There are some helper functions for data cleaning and merging multiple csv files (outputs from the matcher) in `load_csv.py` that can be used. This file combines multiple csv files and creates a data table containing all relevant features with their target ids. This output is used to create the dataset (hdf5). There's also a function to replace no-data values with the most common feature. This function can be used when creating the dataset by setting `imputation=True`.
