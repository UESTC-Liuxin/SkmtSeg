import pandas as pd
import numpy as np
path = '../confusion_matrix.csv'
data = pd.DataFrame(pd.read_csv(path))

print(data.columns)
z=0
for i in data.columns:
    z = z + 1
    if z==1:
        continue
    k=0
    fenmu=np.sum(data[i].values[:11])
    for j in data[i].values:
        if k>10:
            continue
        if str(j) == 'nan':
            print('[{},{},{}]'.format(i, k, 0), end=",")
        else:
            print('[{},{},{:.2f}]'.format(k,i,int(j)/fenmu),end = ",")
        k=k+1
