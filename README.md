# Recommender Systems

This is a framework of the implementation of regularized single-element-based non-negative Matrix-factorization (RSNMF). It was written in python and was tested on MovieLens and Netflix dataset.

Paper link: http://ieeexplore.ieee.org/document/6748996/

MovieLens:
Folder numpy_implement contains codes to perform for loop computation, matrix operation and vector computation of the algorithm. 
    dataset_io.py -> function to load dataset
    rating_matrix.py -> implement the algorithm and grid search

Netflix:
Folder pre_processing is related to the pre-processing of the dataset.
    run.sh -> bash script to perform pre-processing
    load_dataset.py -> merge the original dataset
    user_id_table.py -> save the mapping between original ID and revised ID
    revise_user_id.py -> revise user ID to be in the range of (0, 480189)
    user_list.pickle, user_table.pickle -> user ID table

dataset_io.py -> old version of reading dataset with pytorch
data_block.py -> split the dataset into HDF5 blocks
rating_matrix.py -> implementation in pytorch
recommender.py -> output recommendation list
plot.py -> plot figure of grid search result