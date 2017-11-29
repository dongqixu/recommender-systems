#!/bin/bash
echo 'load dataset'
python load_dataset.py

echo 'get the table'
python user_id_table.py

echo 'revise and output'
python revise_user_id.py

echo 'sort with linux'
sort -n -k1 -k2 dataset_no_sorting.txt -o user_group.txt
sort -n -k2 -k1 dataset_no_sorting.txt -o movie_group.txt
