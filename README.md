The data was imported from https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images

# Pneumonia Detection Using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as either "Normal" or "Pneumonia". The workflow includes data preprocessing, model training, evaluation, and visualization of results.

## Project Structure

- **Data**: Images should be organized in `train`, `val`, and `test` folders.
- **Notebook**: All code is in [notebookd6cd297f58.ipynb](notebookd6cd297f58.ipynb).

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas

Install dependencies:
```sh
pip install torch torchvision scikit-learn matplotlib seaborn pandas
```

## Data Preparation

- Images are loaded using `torchvision.datasets.ImageFolder`.
- Data augmentation and normalization are applied:
    - Resize to 128x128
    - Random horizontal flip and rotation (train only)
    - Normalize using ImageNet mean and std

## Model Architecture

Implemented in [`PneumoniaCNN`](Untitled.ipynb):

- 3 convolutional blocks with batch normalization and max pooling
- Dropout (0.6) for regularization
- 3 fully connected layers
- Output: 2 logits (binary classification)

## Training

- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Training and validation accuracy/loss are tracked and plotted.

## Evaluation

- Confusion matrix and metrics (TP, FP, TN, FN) are computed and visualized.
- ROC curve and AUC are plotted.
- Test accuracy is reported.

## Usage

1. Place your data in `train`, `val`, and `test` folders.
2. Run [notebookd6cd297f58.ipynb](notebookd6cd297f58.ipynb) step by step.
3. Outputs:
    - Training/validation loss and accuracy plots (`training_validation_loss_accuracy.png`)
    - Class distribution plot (`class_distribution.png`)
    - Confusion matrix (`confusion_matrix_with_metrics.png`)
    - ROC curve (`roc_curve_train_test.png`)
    - Model metrics printed in notebook output

## Notes

- The model is defined for PyTorch. The line `model.save('pneumonia_model.h5')` is not valid for PyTorch; use `torch.save(model.state_dict(), 'pneumonia_model.pth')` instead.
- The notebook uses GPU if available.

## Results

- AUC and test accuracy are printed after evaluation.
- AUC: 92.63%,Test Accuracy: 72.92%
- Confusion matrix and ROC curve are saved as images.

## License

This project is
