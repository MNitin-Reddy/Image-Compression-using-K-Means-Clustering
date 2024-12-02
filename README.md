# Recommenders-Reinforcement-Learning-Projects


# K-means Clustering and Image Compression

## 1 - Implementing K-means

The K-means algorithm is a method to automatically cluster similar data points together.

### Algorithm Overview:
1. **Input**: You are given a dataset \( \{x^{(1)}, \dots, x^{(m)}\} \) and want to group the data into \( K \) clusters.
2. **Process**: 
   - **Initialization**: Start by guessing initial centroids.
   - **Iterative Process**: 
     - Assign each data point to the nearest centroid.
     - Recompute centroids by averaging the points assigned to them.
```python
# Initialize centroids
centroids = kMeans_init_centroids(X, K)

for iter in range(iterations):
    # Cluster assignment step:
    idx = find_closest_centroids(X, centroids)

    # Move centroid step:
    centroids = compute_centroids(X, idx, K)
```

### Key Steps:
- **Cluster Assignment**: Assign each data point to its nearest centroid.
- **Centroid Update**: Update each centroid as the mean of the points assigned to it.

---

## 1.1 Finding Closest Centroids

In this phase, the algorithm assigns each training example \( x^{(i)} \) to its closest centroid based on the current positions of centroids.
```python
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids
    """
    
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = np.zeros(K)  # Store distances from the point to each centroid
        for j in range(K):
            # Compute the Euclidean distance between X[i] and the j-th centroid
            distances[j] = np.linalg.norm(X[i] - centroids[j])
        
        # Find the index of the closest centroid
        idx[i] = np.argmin(distances)
    
    return idx
```
---

## 1.2 Computing Centroid Means

After assigning points to centroids, we need to recompute the centroids based on the mean of the points assigned to each centroid.
```python
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray): (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X
        K (int): Number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # Initialize centroids matrix
    centroids = np.zeros((K, n))
    
    for k in range(K):
        # Get all points assigned to the k-th centroid
        points = X[idx == k]
        
        # Compute the mean of those points to find the new centroid position
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    
    return centroids
```
---

## 2 - K-means on a Sample Dataset

Once the `find_closest_centroids` and `compute_centroids` functions are implemented, we can run K-means on a sample 2D dataset.
```python
# Initialize centroids
initial_centroids = kMeans_init_centroids(X, K)

# Run K-means algorithm
for i in range(max_iters):
    idx = find_closest_centroids(X, initial_centroids)
    centroids = compute_centroids(X, idx, K)
```
---

## 3 - Random Initialization

In the K-means algorithm, centroids are often initialized randomly. This helps to avoid the algorithm converging to suboptimal solutions.
```python
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids by selecting K random examples
    from the dataset X.
    
    Args:
        X (ndarray): Data points 
        K (int): Number of centroids
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids
```

---

## 4 - Image Compression with K-means

In this part, we use the K-means algorithm to compress an image by reducing the number of colors used in its representation.
```python

```

### Steps for Image Compression:

1. **Load Image**:
   - Use `matplotlib` to load the image into a three-dimensional matrix.
   - The shape of the matrix will be \( (128, 128, 3) \), where each pixel has three values corresponding to RGB.

2. **Reshape Image**:
   - Convert the image from a 3D matrix into a 2D matrix, where each row represents a pixel with three RGB values.

3. **Run K-means**:
   - Apply the K-means algorithm on the image pixels to reduce the number of colors.
```python
K = 16  # Number of colors
max_iters = 10

initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-means on the image
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
```

4. **Reconstruct Image**:
   - After finding the centroids, reconstruct the image by replacing each pixel with its closest centroid's color.
```python
# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image back into the original dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)
```

5. **Display Original and Compressed Image**:
![image](https://github.com/user-attachments/assets/5696530b-6671-4a77-bfc1-546aaa5ce971)


### Image Compression Results:
- The original image used 24 bits per pixel (3 color channels, 8 bits each).
- The compressed image uses only \( K \times 24 + 128 \times 128 \times 4 \) bits (16 colors, 4 bits per pixel).

This results in a significant reduction in image size, while maintaining most of the visual characteristics of the original image.

---

### Conclusion:
In this project, we implemented the K-means clustering algorithm, explored its application on a sample dataset, and then applied it to compress an image by reducing the number of colors used to represent it. This not only reduced the image's data size but also showcased the practical application of K-means in image processing tasks.
