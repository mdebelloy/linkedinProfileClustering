import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

#read URLs from text file
def getURLS(file):
  file = open(file, 'r')
  return file.read().splitlines()

#reduce embedding dimension from 1536 to 30 using PCA, then 2 using TSNE
def dimensionReduction(dataset):
  #reduce dimensionality to 30 using PCA
  pca = PCA(n_components=30)
  pcaResult = pca.fit_transform(dataset)

  #reduce further to 2 dimensions using TSNE
  tsne = TSNE(n_components=2, verbose=0, perplexity=5, n_iter=3000)
  return tsne.fit_transform(pcaResult)

#plot associated 2d data and apply K-means
def plot(data, annotations):
  #predict the labels of clusters using K-means
  kmeans = KMeans(n_clusters= 3)
  fitData = kmeans.fit_predict(data)

  #plot data using Seaborn and color based on K-means
  sns.scatterplot(
    data = data,
    x=data[:,0], 
    y=data[:,1], 
    hue = fitData,
    palette = "deep"
    )

  plt.title('Clusters of Linkedin Profiles')
  plt.xlabel('TSNE - 1')
  plt.ylabel('TSNE - 2')
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)

  #print numbers of profiles to match datapoints on plot to profiles
  for i in range(len(annotations)):
    print(i + 1, " --> ", annotations[i])
    plt.text(data[i][0] + 5, data[i][1] + 5, i + 1)

  plt.show()

if __name__ == "__main__":
  profileURLS = getURLS("profilesList.txt")

  #open embedded dataset
  dataset = pickle.load( open( "serializedData.txt", "rb" ) )

  plot(dimensionReduction(dataset), profileURLS)