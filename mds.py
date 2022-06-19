import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE


def request(*args, **kwargs):
    src, dst = args[0][0], args[0][1]
    url = f"https://routing.openstreetmap.de/routed-car/route/v1/driving/{src[0]},{src[1]};{dst[0]},{dst[1]}?overview=full"
    # url = f"https://routing.openstreetmap.de/routed-car/route/v1/driving/51.375,35.75;49.69097,34.09694?overview=full"
    payload={}
    headers = {
    'authority': 'routing.openstreetmap.de',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
    'sec-ch-ua-mobile': '?0',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'sec-ch-ua-platform': '"Linux"',
    'accept': '*/*',
    'origin': 'https://payaneha.ir',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://payaneha.ir/',
    'accept-language': 'en-US,en;q=0.9,fa-IR;q=0.8,fa;q=0.7'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    print(f"request from {src[0]},{src[1]}; to {dst[0]},{dst[1]}")
    # print(response.json())
    try:
        return response.json()["routes"][0]["distance"]
    except KeyError:
        return 0

def load(filename="/home/seyed/Documents/geo.csv", is_lightweight=False) -> Tuple[list, dict]:
    df = pd.read_csv(filename)
    state, _, lang, lat, _, _ , _ = df.columns
    df = df[[state, lang, lat]]
    dis = []
    compute = []
    locs = {}
    for index, row in df.iterrows():
        st, lng, lt = row[state], row[lang], row[lat]
        locs[(lng,lt)] = st
        for index_2, row_2 in df.iterrows():
            st, lng_2, lt_2 = row_2[state], row_2[lang], row_2[lat]
            if index == index_2:
                dis.append(((lng,lt), (lng_2, lt_2), 0.0))
            else:
                compute.append(((lng,lt), (lng_2, lt_2)))
                # dis.append(((lng,lt), (lng_2, lt_2), request((lng,lt), (lng_2, lt_2))))
    # import pdb;pdb.set_trace()
    if is_lightweight:
        with ProcessPoolExecutor() as pro:
            res = pro.map(request, compute)
            for i in range(len(compute)):
                dis.append((*compute[i], next(res)))
    else:
        with open("distance.txt", "r") as f:
            ff = f.read()
        dis = list(eval(ff))
    return dis, locs


dis, locs = load()

rows = dict(sorted(locs.items(), key=lambda x: x[1]))
cols = rows.keys()
ind = {col:i for i, col in enumerate(cols)}
data = {col:[0]*len(cols) for col in cols}
for x in dis:
    if x[0] == x[1]:
        data[x[0]][ind[x[1]]] = x[2]
    else:
        if data[x[0]][ind[x[1]]] == 0:
            data[x[0]][ind[x[1]]] = x[2]
            data[x[1]][ind[x[0]]] = x[2]

dist_df = pd.DataFrame.from_dict(data, orient='index',
                       columns=cols)

dist_np = dist_df.to_numpy()

def to_ar(x):
    return get_display(arabic_reshaper.reshape(x))

mds = MDS(dissimilarity="precomputed", random_state=0)


X_transform_isomap = mds.fit_transform(dist_np)

fig = plt.figure(1, (10,4))

size = [64]* len(X_transform_isomap)
ax = fig.add_subplot(122)
plt.scatter(X_transform_isomap[:,0], X_transform_isomap[:,1], s=size)
for j,txt in enumerate(X_transform_isomap[:,0]):
    plt.annotate(to_ar(list(rows.values())[j]), (X_transform_isomap[:,0][j], X_transform_isomap[:,1][j]))
plt.title('MDS')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.show()

knn = [i for i in range(7,16)]
isomaps = [Isomap(knn[i], metric="precomputed") for i in range(len(knn))]


fig = plt.figure(1, (90,81))

for i, isomap in enumerate(isomaps):
    X_transform_isomap = isomap.fit_transform(dist_np)
    size = [64]* len(X_transform_isomap)
    ax = fig.add_subplot(3,3,i+1)
    plt.scatter(X_transform_isomap[:,0], X_transform_isomap[:,1], s=size)
    for j,txt in enumerate(X_transform_isomap[:,0]):
        plt.annotate(to_ar(list(rows.values())[j]), (X_transform_isomap[:,0][j], X_transform_isomap[:,1][j]))
    plt.title(f'Isomap with k = {7+i}')
    fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.show()



spe = SpectralEmbedding(affinity="precomputed", random_state=0)


X_transform_isomap = spe.fit_transform(dist_np)

fig = plt.figure(1, (10,4))

size = [64]* len(X_transform_isomap)
ax = fig.add_subplot(122)
plt.scatter(X_transform_isomap[:,0], X_transform_isomap[:,1], s=size)
for j,txt in enumerate(X_transform_isomap[:,0]):
    plt.annotate(list(rows.values())[j], (X_transform_isomap[:,0][j], X_transform_isomap[:,1][j]))
plt.title('SpectralEmbedding')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.show()



tsne = TSNE(metric="precomputed", random_state=0)


X_transform_isomap = tsne.fit_transform(dist_np)

fig = plt.figure(1, (10,4))

size = [64]* len(X_transform_isomap)
ax = fig.add_subplot(122)
plt.scatter(X_transform_isomap[:,0], X_transform_isomap[:,1], s=size)
for j,txt in enumerate(X_transform_isomap[:,0]):
    plt.annotate(list(rows.values())[j], (X_transform_isomap[:,0][j], X_transform_isomap[:,1][j]))
plt.title('TSNE')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.show()