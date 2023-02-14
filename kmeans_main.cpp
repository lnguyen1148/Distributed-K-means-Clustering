// Distributed K-means Clustering
// Linh Nguyen

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>

#define MAX_ITERATION 100

// This function generates random values and assigns them to data points
void create_data(int numOfPoints, double *points_x, double *points_y)
{
  srand(0);
  for (int i=0; i < numOfPoints; i++)
  {
    points_x[i] = (double) rand() / RAND_MAX;
    points_y[i] = (double) rand() / RAND_MAX;
  }
}

// This functions randomly pick a data point from dataset to assign it to initial centroid
void initialize_centroids(int numOfPoints, int numOfClusters, double *points_x, double *points_y, double *centroids_x, double *centroids_y)
{
  srand(numOfPoints);
  int index;
  for (int i=0; i < numOfClusters; i++)
  {
    index = (int) rand() % numOfPoints + 1;
    centroids_x[i] = points_x[index];
    centroids_y[i] = points_y[index];
  }
}

double euclidean_dist(double pointA_x, double pointA_y, double pointB_x, double pointB_y)
{
  return sqrt(pow(pointA_x - pointB_x, 2) + pow(pointA_y - pointB_y, 2));
}

void k_means_process(int numOfClusters, int subsetSize, double *cluster_x, double *cluster_y, double *centroids_x, double *centroids_y)
{
  double total_x[numOfClusters] = {0};  // total distance from one centroid to points of that cluster
  double total_y[numOfClusters] = {0};
  double min_dist = RAND_MAX;
  double temp_dist = 0;
  int cluster_num;
  int clusterSize[numOfClusters] = {0};

  for (int i=0; i < subsetSize; i++) // iterate datapoints
  {
    for (int j=0; j < numOfClusters; j++) // iterate centroids
    {
      temp_dist = euclidean_dist(cluster_x[i], cluster_y[i], centroids_x[j], centroids_y[j]);
      if (temp_dist < min_dist)
      {
        min_dist = temp_dist;
        cluster_num = j; // cluster number corresponding to the minimum distance
        clusterSize[j] += 1;
      }
    }
    total_x[cluster_num] += cluster_x[i];
    total_y[cluster_num] += cluster_y[i];
  }

  // Update new centroids at the current process
  for (int j=0; j < numOfClusters; j++)
  {
    centroids_x[j] = total_x[cluster_num] / clusterSize[j];
    centroids_y[j] = total_y[cluster_num] / clusterSize[j];
  }
}

int main( int argc, char *argv[] )
{
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
 
  double *centroids_x;
  double *centroids_y;
  double *points_x;
  double *points_y;
  double *cluster_x;
  double *cluster_y;
  double *centroid_reduce_x;
  double *centroid_reduce_y;

  int numOfProcesses = atoi(argv[1]);
  int numOfClusters = atoi(argv[2]);
  int numOfPoints = atoi(argv[3]);

  int subsetSize = numOfPoints / numOfProcesses + 1;

  cluster_x = (double*) malloc(sizeof(double) * subsetSize);
  cluster_y = (double*) malloc(sizeof(double) * subsetSize);

  points_x = (double*) malloc(sizeof(double) * numOfPoints);
  points_y = (double*) malloc(sizeof(double) * numOfPoints);

  centroids_x = (double*) malloc(sizeof(double) * numOfClusters);
  centroids_y = (double*) malloc(sizeof(double) * numOfClusters);

  centroid_reduce_x = (double*) malloc(sizeof(double) * numOfClusters);
  centroid_reduce_y = (double*) malloc(sizeof(double) * numOfClusters);

  if (rank == 0)
  {
    // Create dataset by assigning random values
    create_data(numOfPoints, points_x, points_y);

    // Randomly pick datapoints as initial centroids
    initialize_centroids(numOfPoints, numOfClusters, points_x, points_y, centroids_x, centroids_y);
  }

  // Broadcast centroids to all processes
  MPI_Bcast(centroids_x, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(centroids_y, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Scatter subsets to processes
  MPI_Scatter(points_x, subsetSize, MPI_DOUBLE, cluster_x, subsetSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(points_y, subsetSize, MPI_DOUBLE, cluster_y, subsetSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
  int iter = 0;
  while (iter < MAX_ITERATION)
  {
    k_means_process(numOfClusters, subsetSize, cluster_x, cluster_y, centroids_x, centroids_y);
    MPI_Barrier(MPI_COMM_WORLD);  // wait for all processes to complete the calculations

    // Take the sum of centroids from processes to calculate the mean
    MPI_Reduce(centroids_x, centroid_reduce_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(centroids_y, centroid_reduce_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)  // At root process, update centroids
    {
      for (int i=0; i < numOfClusters; i++)
      {
        centroids_x[i] = centroid_reduce_x[i] / numOfProcesses;
        centroids_y[i] = centroid_reduce_y[i] / numOfProcesses;
      }
    }

    // Broadcast new centroids to all processes
    MPI_Bcast(centroids_x, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(centroids_y, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    iter++;
  }

  // After all processes complete k-means algorithm, we can print out the result
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    std::cout << "K-means clustering for " << numOfPoints << " points of " << numOfClusters << " clusters on " << numOfProcesses << " processes..." << std::endl;
    std::cout <<  "After " << MAX_ITERATION << " iterations:" << std::endl;
    for (int i=0; i < numOfClusters; i++)
    {
      std::cout << "Centroid of cluster " << i << ": x=" << centroids_x[i] << ", y=" << centroids_y[i] << std::endl;
    }
  }

  free(points_x);
  free(points_y);
  free(centroids_x);
  free(centroids_y);
  free(cluster_x);
  free(cluster_y);
  free(centroid_reduce_x);
  free(centroid_reduce_y);

  MPI_Finalize();
  return 0;
}
