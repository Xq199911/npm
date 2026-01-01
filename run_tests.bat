@echo off
call conda activate py
set PYTHONPATH=.
python tests/test_clustering.py
pause
