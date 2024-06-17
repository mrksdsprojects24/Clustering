import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Initial parameters
n_samples = 200
n_features = 2
n_clusters = None  # Placeholder for user-defined number of clusters

# Function to generate synthetic data
def generate_data(n_blobs, random_state):
  return make_blobs(n_samples=n_samples, n_features=n_features, centers=n_blobs, random_state=random_state)

# Streamlit app layout
st.title("Krishna's Playground for Clustering Algorithms")

st.write("Please select the number of blobs to generate and a random value between 0 and 100. If you do not like the data generated, you can always play with the random seed. Sometimes a couple of blobs get merged. You do not have worry about it. You may choose the number of clusters based on what you see in the data generated. Whichever algorithm gives higher Silhouette score is the winner!")
# User input for number of blobs and random seed
n_blobs = st.number_input("Number of Blobs", min_value=1, max_value=10, value=3)
random_state = st.number_input("Random Seed", min_value=0, value=42)

# Generate data based on user input
X, _ = generate_data(n_blobs, random_state)

# Display generated data as a scatter plot
st.subheader("Generated Data")
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])
st.pyplot(fig)

# User input for number of clusters and algorithm(s) to run
st.write("Please select the number of clusters and which clustering algorithm(s) to run.")
n_clusters = st.number_input("Number of Clusters:", min_value=1, value=n_blobs)
kmeans_run = st.checkbox("Run K-means")
gmm_run = st.checkbox("Run GMM")

if kmeans_run or gmm_run:
  # Run clustering algorithms if user selects either
  if n_clusters is None:
    n_clusters = n_blobs  # Default to number of blobs for clusters

  # K-means clustering
  if kmeans_run:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Fixed random state for KMeans
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_
    kmeans_silhouette_score = silhouette_score(X, kmeans_labels)

  # GMM clustering
  if gmm_run:
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)  # Fixed random state for GMM
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    gmm_silhouette_score = silhouette_score(X, gmm_labels)

  # Display clustering results if algorithms were run
  if kmeans_run and gmm_run:
    # Display results in two columns
    st.subheader("Clustering Results")
    col1, col2 = st.columns(2)
    with col1:
      st.write("**K-means Clustering**")
      fig, ax = plt.subplots()
      ax.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
      st.pyplot(fig)
#      st.pyplot(X, c=kmeans_labels)
      st.write(f"Silhouette Score: {kmeans_silhouette_score:.3f}")
    with col2:
      st.write("**GMM Clustering**")
      fig, ax = plt.subplots()
      ax.scatter(X[:, 0], X[:, 1], c=gmm_labels)
      st.pyplot(fig)
#      st.pyplot(X, c=gmm_labels)
      st.write(f"Silhouette Score: {gmm_silhouette_score:.3f}")
  else:
    # Display results for the selected algorithm
    if kmeans_run:
      st.subheader("K-means Clustering Results")
      fig, ax = plt.subplots()
      ax.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
      st.pyplot(fig)
#      st.pyplot(X, c=kmeans_labels)
      st.write(f"Silhouette Score: {kmeans_silhouette_score:.3f}")
    else:
      st.subheader("GMM Clustering Results")
      fig, ax = plt.subplots()
      ax.scatter(X[:, 0], X[:, 1], c=gmm_labels)
      st.pyplot(fig)
#      st.pyplot(X, c=gmm_labels)
      st.write(f"Silhouette Score: {kmeans_silhouette_score:.3f}")
