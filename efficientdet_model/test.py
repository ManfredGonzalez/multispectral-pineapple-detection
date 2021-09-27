import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#print(os.getcwd())
print(f'{os.path.dirname(os.path.abspath(__file__))}')
#print(os.path.exists(f'{os.path.dirname(os.path.abspath(__file__))}/datasets'))
#print(os.path.dirname(os.path.abspath(__file__)))
print(os.path.exists('datasets/test'))
#print(os.listdir('datasets/'))