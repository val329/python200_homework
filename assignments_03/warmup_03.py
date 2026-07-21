import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# --------------------Preprocessing-------------------- #
# Q1
# Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42. 
# Print the shapes of all four arrays.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Q2
# Fit a StandardScaler on X_train and use it to transform both X_train and X_test. 
# Print the mean of each column in X_train_scaled -- they should all be very close to 0. 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print(X_train_scaled.mean(axis=0))              # axis=0 -> mean for every column
# fitting the scaler on X_train only to avoid data leakage (test set influencing overall statistics)


# --------------------KNN-------------------- #
# Q1
# Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), 
# and predict on the test set. Print the accuracy score and the full classification report.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


# Q2
# Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). Print the accuracy score. 
# Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for 
# this particular dataset?
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

predictions = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
# Scaling does not seem to make a difference. All the measurements were taken in the same units. Also the sample
# size is small with a random split, so high chance of perfect case with no errors


# Q3
# Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. Print each fold score, 
# the mean, and the standard deviation. Add a comment: is this result more or less trustworthy than a single train/test 
# split, and why?
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(cv_scores)           # accuracy on each fold
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# This result can be considered more trustworthy than a single split because every training example participates in 
# evaluation, and the average score is more stable


# Q4
# Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each, compute 5-fold cross-validation accuracy on the unscaled 
# training data and print k and the mean CV score. Add a comment identifying which k you would choose and why.
k_values = [1, 3, 5, 7, 9, 11, 13, 15] 

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

# k with the highest mean and lowest std is k= 7 (mean=0.975  std=0.020)


# --------------------Classifier Evaluation-------------------- #
# Q1
# Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, 
# passing display_labels=iris.target_names. Save the figure to outputs/knn_confusion_matrix.png. Add a comment: 
# which pair of species does the model most often confuse (if any)?
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig("outputs/knn_confusion_matrix.png")


# --------------------Logistic Regression and Regularization-------------------- #
# Q1
# Train three logistic regression models on the scaled Iris data, identical in every way except for the 
# C parameter: C=0.01, C=1.0, and C=100. Use max_iter=1000 and solver='liblinear' for all three. For each 
# model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum(). 
# Add a comment: what happens to the total coefficient magnitude as C increases? What does this tell 
# you about what regularization is doing?

log_reg_1 = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    C=0.01
)
log_reg_1.fit(X_train_scaled, y_train)
print("C=0.01", np.abs(log_reg_1.coef_).sum())

log_reg_2 = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    C=1.0
)
log_reg_2.fit(X_train_scaled, y_train)
print("C=1.0", np.abs(log_reg_2.coef_).sum())

log_reg_3 = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    C=100
)
log_reg_3.fit(X_train_scaled, y_train)
print("C=100", np.abs(log_reg_3.coef_).sum())




# --------------------PCA-------------------- #
# import gdown
# from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


# Q1
# Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), 
# using cmap='gray_r' with each digit's label as the title. Save the figure to outputs/sample_digits.png. 
# (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, so digits appear as 
# dark ink on a light background, which is more readable than the default.)
print(X_digits.shape, images.shape)
num_frames, num_rows, num_cols = images.shape

# Find one example index for each digit 0–9
example_indices = [np.where(digits.target == d)[0][0] for d in range(10)]

plt.figure(figsize=(12, 3))

for i, idx in enumerate(example_indices):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[idx], cmap='gray_r')
    plt.title(str(i))
    plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")


# Q2
# Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits). 
# As in the lesson, scores tell you how strongly each component is weighted for each sample -- scores[i, 0] is 
# the weighting for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.
# Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar. 
# Here is the pattern for coloring by a label array and attaching a colorbar:
# scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
# plt.colorbar(scatter, label='Digit')
# Save the figure to outputs/pca_2d_projection.png. Add a comment: do same-digit images tend to cluster together in this 2D space?

X_digits = images.reshape(num_frames, -1).astype(np.float32)  # num_samples x num_dimensions

pca = PCA(n_components=4, svd_solver="randomized", random_state=0)
pca.fit(X_digits)

scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig("outputs/pca_2d_projection.png")

# Q3
# Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using 
# np.cumsum(pca.explained_variance_ratio_). Save to outputs/pca_variance_explained.png. Add a comment: approximately 
# how many components do you need to explain 80% of the variance?
perc_exp_vals = pca.explained_variance_ratio_ * 100
total_explained = perc_exp_vals.sum()
print("Explained variance (%):", ", ".join(f"{v:.2f}" for v in perc_exp_vals))
print(f"Total (%): {total_explained:.2f}")


# Q4
# The preprocessing lesson showed that a reconstruction is built by starting from the mean and adding each component 
# weighted by its score. Here is the same idea generalized to n components -- add this function to your file:

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits 
# using reconstruction through principal components n = 2, 5, 15, and 40.
# Build a grid of subplots where rows correspond to each n value and columns show those 5 digits. Add an "Original" row at the top (use images[i], which is already shaped as (8, 8)). Save to outputs/pca_reconstructions.png.
# Add a comment: at what n do the digits become clearly recognizable, and does that match where the variance curve levels off?
