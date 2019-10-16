import pandas as pd
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns

import requests
import time
from multiprocessing import Pool

# df = pd.read_csv('hdb-property-information.csv')
# print(df.columns)
#
# building_df = pd.read_json('buildings.json')
# print(building_df.columns)
#
#
#
# important_columns = ['street','market_hawker','miscellaneous','total_dwelling_units']
# housing_df = df[important_columns]
#
#
# streetname = housing_df['street']
# markets = housing_df['market_hawker']
# #Examples include admin office, childcare centre, education centre, Residents' Committees centre
# misc = housing_df['miscellaneous']
# population = housing_df['total_dwelling_units']
#
#
# housing_df.replace('Y',1,inplace=True)
# housing_df.replace('N',0,inplace=True)
#
#
# print(housing_df.describe())
#
# print(housing_df.groupby('street').sum().head())


shp_path = "./Bollard_Apr2019/Bollard.shp"
sf = shp.Reader(shp_path)

sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.mpl.rc("figure", figsize=(100,60))



def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords'
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

df = read_shapefile(sf)


def plot_map(sf, x_lim=None, y_lim=None, figsize=(11, 9)):
    '''
    Plot map with lim coordinates
    '''
    plt.figure(figsize=figsize)
    id = 0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k')

        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=10)
        id = id + 1

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

plot_map(sf)
plt.show()





