## Project: Implement a distributed algorithm using MPI
### Distributed K-Means Clustering


#### Running the program: 
   Run these two commands: 	

- mpic++ kmeans_main.cpp -o kmeans
- mpirun -n p kmeans p c d
    
(With	p: number of processes,
c: number of clusters,
d: number of datapoints)

Example: Run 3 processes, 5 clusters, 60 datapoints:
- mpic++ kmeans_main.cpp -o kmeans
- mpirun -n 3 kmeans 3 5 60
