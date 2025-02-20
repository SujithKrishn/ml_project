import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample dataset: Customer annual income, spending score, and age
data = {
    'AnnualIncome': [
        15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
        35,   # Normal points
        80, 85, 90  # Outliers
    ],
    'SpendingScore': [
        39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
        69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
        6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
        5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
        56,   # Normal points
        2, 3, 100  # Outliers
    ],
    'Age': [
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
        35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5,
        40,   # Normal points
        15, 60, 70  # Outliers
    ]
}
df = pd.DataFrame(data)
print(df.head())

# Normalize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Convert back to DataFrame for easy handling
df_scaled = pd.DataFrame(scaled, columns=['AnnualIncome', 'SpendingScore','Age'])

# Plotting the 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], df_scaled['Age'], c='blue', marker='o')

# Adding labels
ax.set_xlabel('Annual Income (scaled)')
ax.set_ylabel('Spending Score (scaled)')
ax.set_zlabel('Age (scaled)')

# Show plot
plt.show()


from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions from 3 to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(scaled)

# Convert back to DataFrame for easy handling
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())

import matplotlib.pyplot as plt

# Plot the PCA components
plt.scatter(df_pca['PCA1'], df_pca['PCA2'])
plt.title('PCA - Dimensionality Reduction')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


from sklearn.manifold import TSNE

# Apply t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
df_tsne = tsne.fit_transform(scaled)

# Convert to DataFrame
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print(df_tsne.head())


# Plot the t-SNE components
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'])
plt.title('t-SNE - Dimensionality Reduction')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()