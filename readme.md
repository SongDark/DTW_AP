# DTW based Affinity Propagation Clustering

AP Clustering using DTW distance for temporal sequences classification.

## CharacterTrajectory

### Time Consumption
`~` means "about"

N| N(N-1)/2| np_dtw (r=10) | np_dtw_parallel (r=10, cpu=15) |tf_dtw (batches) |
:---: | :---:| :---: | :---: | :---:
10 | 45 | 8.07s | 1.29s | 16.31s (1)
50 | 1,225 | 229.88 | 26.60 |72.30 (5)
100 | 4,950 | 959.57 | 102.74 |72.97 (5)
200 |19,900 |  ~4k (1.1h) | 402.69 |77.13 (5)
1,000 | 499,500 | ~100k (27h) | ~10k (2.8h) |795.20 (50)
2,858 | 4,082,653 | ~784k (9d) | ~78.4k (21h) |6476.16 (400)

### DTW Matrix

<img src="save/CharacterTrajectories/CharacterTrajectories_dtwmat_tf.png">

### Confusion Matrix

<img src="save/CharacterTrajectories/CharacterTrajectories_confusion_matrix.png">

