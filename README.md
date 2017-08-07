# ml4seti - Winning Entry

This is the repository for the winning model entry in the [___ml4seti___ __Code Challenge/Competition__](http://www.seti.org/ml4seti), organized by the [___Search for ExtraTerrestrial Intelligence (SETI) Institute___](www.seti.org).

The objective of the _ml4seti_ was to train a classifier to differentiate between the following signal types:
* brightpixel
* narrowband
* narrowbanddrd
* noise
* squarepulsednarrowband
* squiggle
* squigglesquarepulsednarrowband

Check out the [_ml4seti_ Github](https://github.com/setiQuest/ML4SETI) for more details.

The criterion to select the winning entry was the (multinomial) _LogLoss_ of the model. 

---
## Data

The training data contained 140,000 signals across the 7 classes. Our data preparation pipeline involved:

1. Creating 5 stratified (equal class distribution) folds from the training data. 
2. The complex valued time series of each signal is divided into 384 chunks of 512 timesteps each.
3. Hann Windows are applied. 
4. Fast Fourier Transform (FFT) is performed on the chunks.
5. Two features are generated from the complex amplitudes -
    - Log of Square of Absolute Value of Amplitude
    - Phase
6. Each resulting [384, 512, 2] tensor is normalized by the frequency-bin-wise mean and standard deviation across the entire training dataset.

All tensors were stored in HDF5 files, and batches are read directly from disk during model training.

---
## Model

For the winning entry, we used an averaged ensemble of 5 [___Wide Residual Networks___](https://arxiv.org/abs/1605.07146), trained on different sets of 4(/5) folds, each with a depth of 34 (convolutional layers) and a widening factor of 2.

![WResnet34x2 Architecture](https://github.com/sgrvinod/Effsubsee-ml4seti-Code-Challenge/blob/master/img/wresnet34x2.png)

In the above figure, the architecture of each _BasicBlock_ is as follows -- 

![BasicBlock Architecture](https://github.com/sgrvinod/Effsubsee-ml4seti-Code-Challenge/blob/master/img/basicblock.PNG)

In the interest of time, we used a batch size of 96 and an aggressive learning rate schedule. Starting at 0.1, we halved the learning rate when the model did not improve for 3 consecutive epochs, and terminated training when it failed to improve for 8 consecutive epochs.

The validation accuracies on each of the folds (having trained on the other 4 folds) is as follows:

| Validation Fold | Validation Accuracy |
| :-------------: | :-----------------: |
| Fold 1 | 95.88% |
| Fold 2 | 95.74% |
| Fold 3 | 95.79% |
| Fold 4 | 95.65% |
| Fold 5 | 95.77% |

For the final submission, each of these 5 fold-models were run on the test dataset, and their scores averaged.

---

## Evaluating the model

Dependencies:
`pytorch 0.1.12`
`torchvision 0.1.8`
`pandas`
`h5py`
`scikit-learn`

To evaluate the model(s) and/or to reproduce our results on the Final Test Set:

1. Download the signal files and the corresponding CSV with the `UUID` column into a single folder.

2. In `./folds/create_h5_tensors.py`, run the `create_test_tensors_hdf5_logmod2_ph()` function, while pointing to your raw signal data. This will create an hdf5 file with the test tensor.

3. Run `test.py` with the architecture name, checkpoint, test hdf5 file, and the hdf5 file containing the mean and standard deviation used for normalizing. Do this for all 5 folds. For example, for fold 1:

    `python test.py 'wresnet34x2' './wresnet34x2 models/wresnet34x2 FOLD1/FOLD1_BEST_wresnet34x2_batchsize96_checkpoint.pth' 'path/to/your/test/hdf5' './folds/mean_stddev_primary_full_v3__384t__512f__logmod2-ph.hdf5'`.

    The CSVs with the scores for each fold-model will be saved in the same folder as `test.py`.
    
    Note: If you don't have a CUDA-enabled GPU, the code will need to be modified to support CPU. Reach out to us on the _ml4seti_ Slack channel if you need help with this.

4. Move these CSVs to a separate folder, and run `average_scores.py` by pointing to this folder, and specifying the path for the output CSV:

    `python average_scores.py 'path/to/folder/with/individual/model/scores' 'path/to/output/csv.csv'`

---
(Team) ___Effsubsee___
* Stephane Egly
* Sagar Vinodababu
* Jeffrey Voien
