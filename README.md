# Effsubsee at _ml4seti_ Code Challenge

This is the repository of ___Effsubsee___, the winner of the [___ml4seti___ __Code Challenge__](http://www.seti.org/ml4seti), organized by the [___Search for ExtraTerrestrial Intelligence (SETI) Institute___](www.seti.org).

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

Here are the final LogLoss measures of the top teams on the Final test set:

![Final Test Leaderboard](https://github.com/sgrvinod/Effsubsee-ml4seti-Code-Challenge/blob/master/img/final_leaderboard.png)

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

All tensors were stored in HDF5 files for reading during model training.

---
## Model

For the winning entry, we used an averaged ensemble of 5 [___Wide Residual Networks___](https://arxiv.org/abs/1605.07146), trained on different sets of 4/5 folds, each with a depth of 34 (convolutional layers) and a widening factor of 2:

![WResnet34x2 Architecture](https://github.com/sgrvinod/Effsubsee-ml4seti-Code-Challenge/blob/master/img/wresnet34x2.PNG)

In the above figure, the architecture of each _BasicBlock_ is as follows:

![BasicBlock Architecture](https://github.com/sgrvinod/Effsubsee-ml4seti-Code-Challenge/blob/master/img/basicblock.PNG)

In the interest of time, we used an aggressive Learning Rate schedule. Starting at 0.1, we halved it when the model did not improve for 3 consecutive epochs, and terminated training when it failed to improve for 8 consecutive epochs.

The validation accuracies on each of the folds (having trained on the other 4 folds) is as follows:

| Validation Fold | Validation Accuracy |
| :-------------: | :-----------------: |
| Fold 1 | 95.88% |
| Fold 2 | 95.88% |
| Fold 3 | 95.88% |
| Fold 4 | 95.88% |
| Fold 5 | 95.77% |

For the final submission, each of these 5 fold-models were run on the test dataset, and their scores averaged.

---

(Team) ___Effsubsee___
* Stephane Egly
* Sagar Vinodababu
* Jeffrey Voien
