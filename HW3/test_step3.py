import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PHOTO_DIR = "C:/Users/HP/Desktop/TN10"  # file with picture
IMAGE_SIZE = (128, 128)
EPSILON_LEVELS = [10, 1, 0.1]  # value of confidentiality
NOISE_LEVELS = [5.0, 15.0, 30.0]  # level of noise

# --- upload pictures ---
def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                filenames.append(filename)
    return images, filenames

# ---  PCA + noise gaussien (DP) ---
def apply_differential_privacy(img, noise_level):
    h, w = img.shape
    flat_img = img.flatten().reshape(1, -1)
    pca = PCA(n_components=0.3)  # compression
    compressed = pca.fit_transform(flat_img)
    noise = np.random.normal(0, noise_level, compressed.shape)
    compressed_noisy = compressed + noise
    reconstructed = pca.inverse_transform(compressed_noisy)
    reconstructed_img = reconstructed.reshape(h, w)
    return np.clip(reconstructed_img, 0, 255).astype(np.uint8)

# --- calcul MSE and SSIM ---
def compute_metrics(original, reconstructed):
    mse_val = mean_squared_error(original, reconstructed)
    ssim_val = ssim(original, reconstructed)
    return mse_val, ssim_val

# --- experiment ---
def evaluate_differential_privacy():
    images, names = load_images(PHOTO_DIR)
    results = []

    for eps, noise_std in zip(EPSILON_LEVELS, NOISE_LEVELS):
        for img, name in zip(images, names):
            obfuscated = apply_differential_privacy(img, noise_std)
            mse_val, ssim_val = compute_metrics(img, obfuscated)

            results.append({
                "Filename": name,
                "Epsilon": eps,
                "Noise_STD": noise_std,
                "MSE": mse_val,
                "SSIM": ssim_val
            })

    return pd.DataFrame(results)

# --- print ---
df_results = evaluate_differential_privacy()
print(df_results)

# epsilon average
summary = df_results.groupby("Epsilon").agg({"MSE": "mean", "SSIM": "mean"}).reset_index()
print("\n--- Résumé par niveau de confidentialité (𝜖) ---")
print(summary)

# --- print ---
sns.set(style="whitegrid")
plt.figure(figsize=(10, 4))

# MSE plot
plt.subplot(1, 2, 1)
sns.lineplot(data=summary, x="Epsilon", y="MSE", marker="o")
plt.title("MSE vs Epsilon")
plt.xlabel("Epsilon (𝜖)")
plt.ylabel("Mean Squared Error")

# SSIM plot
plt.subplot(1, 2, 2)
sns.lineplot(data=summary, x="Epsilon", y="SSIM", marker="o", color="green")
plt.title("SSIM vs Epsilon")
plt.xlabel("Epsilon (𝜖)")
plt.ylabel("Structural Similarity Index")

plt.tight_layout()
plt.show()
