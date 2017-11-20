# Recommender-System
Recommender-System

1. How to store and read the matrix?
2. How to improve the calculation speed?
3. Why matrix computation is not working?
4. How many epoch should be trained?

load_dataset.py -> load the original data to one file
user_id_table -> create a user id conversion table
revise_user_id -> change the user id to short form

sh:

echo 'load dataset'
python load_dataset.py

echo 'get the table'
python user_id_table.py

echo 'revise and output'
python revise_user_id.py

echo 'sort'
sort -n -k1 -k2 dataset_no_sorting.txt -o user_group.txt
sort -n -k2 -k1 dataset_no_sorting.txt -o movie_group.txt