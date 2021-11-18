import pandas as pd
import pickle
import csv

df = pd.read_csv('/data/dekim/word-embeddings/glove/glove.txt', sep =" ", quoting=3, header=None, index_col=0)
print("glove file loaded!")
glove = {key: val.values for key, val in df.T.items()}

with open('glove.korean.pkl', 'wb') as fp:
    pickle.dump(glove, fp)