#!/bin/bash
FOLDER=$1
pwd
ls
cd home/
ls
cd ..
cd ..

cd opt/source-code/
pwd 
ls

# cd teste2/
# pwd 
# ls
echo Running script
python3 -u $FOLDER/run.py | tee results/log.txt
pwd
ls
cd results
pwd
ls
