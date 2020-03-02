#!/bin/bash
pwd
cd opt/source-code/
# pwd 
# ls

cd teste2/
# pwd 
# ls
echo Running script
python3 -u run.py | tee results/log.txt
pwd
ls
cd results
pwd
ls
