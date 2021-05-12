class KMeans:
  def __init__(self, initial_clusters, data):
    self.clusters = initial_clusters
    self.data = data
    self.centers = self.get_centers()

  def get_centers(self):
    data_dim = len(self.data[0])
    center_dict = {k:[0 for i in range(data_dim)] for k in self.clusters}
    for key, cluster in self.clusters.items():
      for index in cluster:
        for i in range(data_dim):
          center_dict[key][i] += self.data[index][i] / len(cluster)
    return center_dict

  def calc_dist(self, point1, point2):
    dist = 0
    for i in range(len(point1)):
      dist += (point1[i] - point2[i])**2
    return dist**0.5

  def update_clusters_once(self):
    new_clusters = {k:[] for k in self.clusters}
    data_points = len(self.data)
    # print(self.centers)
    for index in range(data_points):
      dists = {k:self.calc_dist(self.data[index], self.centers[k]) for k in self.clusters}
      new_cluster = min(dists, key=lambda k: dists[k]) 
      # print(new_cluster)
      # print(dists)
      new_clusters[new_cluster].append(index)
    self.clusters = new_clusters
    self.centers = self.get_centers()

  def run(self):
    previous_clusters = {}
    while previous_clusters != self.clusters:
      previous_clusters = self.clusters.copy()
      self.update_clusters_once()
      
