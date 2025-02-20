"""Use case overview
For a business trying to understand customer behavior, unsupervised learning can be applied to segment customers based on their purchasing habits. The company wants to group similar customers together to better target marketing strategies, but they do not have predefined labels for these customer segments.

Data
Input features (X): number of purchases, total spending, product categories purchased

Solution approach
In this case, we can use a clustering algorithm like k-means to group customers based on the similarity of their behaviors. The algorithm will learn to segment customers without needing predefined labels.

Steps involved
Data collection: gather customer purchasing data, such as the number of purchases, spending amount, and categories purchased.

Data preprocessing: standardize the data to ensure that features such as spending and purchases are on similar scales.

Model training: use the k-means algorithm to create clusters of customers with similar behavior.

Cluster analysis: analyze the resulting clusters to understand the common characteristics of customers within each group."""

from sklearn.cluster import KMeans
import numpy as np

# Sample customer data (number of purchases, total spending, product categories)
X = np.array([[5, 1000, 2], [10, 5000, 5], [2, 500, 1], [8, 3000, 3]])

# Create and fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Print the cluster centers and labels
print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")


"""After clustering, the customers are grouped into distinct segments, allowing the business to target marketing efforts more effectively. For example, one cluster may represent high-spending, frequent buyers, while another represents lower-spending, infrequent buyers. Unsupervised learning is appropriate here because there are no predefined labels for customer segments."""