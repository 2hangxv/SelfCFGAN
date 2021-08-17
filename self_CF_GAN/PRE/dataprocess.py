import pandas as pd
import numpy as np

ratings = pd.read_csv('data/ml-100k/u.data', header=None, sep='\t', encoding='latin-1')
n_users = ratings[0].unique().shape[0]#943
n_items = ratings[1].unique().shape[0]#1682


fun_matrix = np.full((n_users, n_items), np.nan)
satis_matrix = np.full((n_users, n_items), np.nan)
for line in ratings.itertuples():
    fun_matrix[line[1]-1, line[2]-1] = 1
    satis_matrix[line[1]-1, line[2]-1] = line[3]

np.save("raw_rating.npy",satis_matrix)
np.save("raw_fun.npy",fun_matrix)
def wgn(x, snr):
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise

for i in range(n_users):
    avg=np.nanmean(satis_matrix[i])
    ST=np.count_nonzero(np.isnan(satis_matrix[i]))
    FT=np.count_nonzero(np.isnan(satis_matrix[i]))
    sx = np.zeros((1, ST)) + 0.5
    fx = np.zeros((1, FT)) + 0.5
    sx_noise = wgn(sx, 50)
    fx_noise = wgn(fx, 50)
    sk=0
    fk=0
    for j in range(n_items):
        if(np.isnan(satis_matrix[i][j])):
            satis_matrix[i][j]=sx_noise[0][sk]
            sk=sk+1
        else:
            satis_matrix[i][j]=1 if satis_matrix[i][j]>avg else 0
        if(np.isnan(fun_matrix[i][j])):
            fun_matrix[i][j]=fx_noise[0][fk]
            fk=fk+1


np.save("fun_matrix.npy",fun_matrix)
np.save("satis_matrix.npy",satis_matrix)