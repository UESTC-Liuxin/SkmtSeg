import pandas as pd
import numpy as np
path = 'confusion/confusion_matrix.csv'
data = pd.DataFrame(pd.read_csv(path))

z=0
fenmu=np.zeros(11)
for i in data.columns:
    z = z + 1
    if z==1:
        continue
    k=0
    for j in data[i].values:
        if k > 10:
            continue
        if str(j) == 'nan':
            fenmu[k] +=0
        else:
            fenmu[k] +=j
        k+=1
z=0
print(fenmu)
for i in data.columns:
    z = z + 1
    if z==1:
        continue
    k=0
    for j in data[i].values:
        if k>10:
            continue
        if str(j) == 'nan':
            print('[{},{},{}]'.format(i,10-k, 0), end=",")
        else:
            print('[{},{},{:.2f}]'.format(i,10-k,int(j)/fenmu[k]),end = ",")
        k=k+1
##列和
# for i in data.columns:
#     z = z + 1
#     if z==1:
#         continue
#     k=0
#     fenmu=np.sum(data[i].values[:11])
#     print(fenmu)
#     for j in data[i].values:
#         if k>10:
#             continue
#         if str(j) == 'nan':
#             print('[{},{},{}]'.format(i, k, 0), end=",")
#         else:
#             print('[{},{},{:.2f}]'.format(k,i,int(j)/fenmu),end = ",")
#         k=k+1
