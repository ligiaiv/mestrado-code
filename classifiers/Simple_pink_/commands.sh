#!/bin/bash

FILE=$1
echo file "$@"
pwd
#ls
# cd home/
# ls
# cd ..
# cd ..
echo "getting in folder"
cd opt/source-code/
pwd 
ls

# cd teste2/
# pwd 
# ls
echo Running script
python3 -u Simple_pink_/$FILE.py 
# echo Finished prep_data
# python3 -u main.py
# | tee results/log.txt
# pwd
# ls
# cd results
# pwd
# ls
