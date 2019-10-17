import pandas as pd
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import geopandas as gpd

import requests
import time
from multiprocessing import Pool

#%%



import pandas as pd
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import geopandas as gpd

import requests
import time
from multiprocessing import Pool

df = pd.read_csv('hdb-property-information.csv')
print(df.columns)

important_housing_columns = ['blk_no','street','market_hawker','miscellaneous','total_dwelling_units']
print(df.columns)
housing_df = df[important_housing_columns]
print(len(housing_df))
print(housing_df.head())

df_2 = pd.read_json('buildings.json')
important_building_columns = ['BLK_NO','ROAD_NAME','LATITUDE','LONGTITUDE']
print(df_2.columns)
building_df = df_2[important_building_columns].rename(columns={'BLK_NO':'blk_no'}).rename(columns={'ROAD_NAME':'street'})
print(len(building_df))
print(building_df.columns)
print(building_df.head())


combined = housing_df.merge(building_df,on=['blk_no','street'],how='inner')
print(combined.columns)



housing_df.replace('Y',1,inplace=True)
housing_df.replace('N',0,inplace=True)


print(housing_df.describe())

print(housing_df.groupby('street').sum().head())


shp_path = "./dwelling/PLAN_BDY_DWELLING_TYPE_2014.shp"
sf = shp.Reader(shp_path)

fig,ax = plt.subplots(figsize=(15,15))


sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.mpl.rc("figure", figsize=(10,6))



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


def plot_map(sf, x_lim=None, y_lim=None, figsize=(11, 6)):
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
            plt.text(x0, y0, id, fontsize=1)
        id = id + 1

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

plot_map(sf)


def plot_map_fill(id, sf, x_lim=None,
                  y_lim=None,
                  figsize=(11, 9),
                  color='r'):
    '''
    Plot map with lim coordinates
    '''

    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')

    shape_ex = sf.shape(id)
    x_lon = np.zeros((len(shape_ex.points), 1))
    y_lat = np.zeros((len(shape_ex.points), 1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    ax.fill(x_lon, y_lat, color)

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)


print('hi')
plot_map_fill(0, sf, color='g')
plt.show()


def plot_map_fill_multiples_ids(title, comuna, sf,
                                x_lim=None,
                                y_lim=None,
                                figsize=(11, 9),
                                color='r'):
    '''
    Plot map with lim coordinates
    '''

    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')

    for id in comuna:
        shape_ex = sf.shape(id)
        x_lon = np.zeros((len(shape_ex.points), 1))
        y_lat = np.zeros((len(shape_ex.points), 1))
        for ip in range(len(shape_ex.points)):
            x_lon[ip] = shape_ex.points[ip][0]
            y_lat[ip] = shape_ex.points[ip][1]
        ax.fill(x_lon, y_lat, color)

        x0 = np.mean(x_lon)
        y0 = np.mean(y_lat)
        plt.text(x0, y0, id, fontsize=10)

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)


comuna_id = [0, 1, 2, 3, 4, 5, 6]
plot_map_fill_multiples_ids("Multiple Shapes",
                            comuna_id, sf, color='r')
plt.show()

def calc_color(data, color=None):
    if color == 1: color_sq = ['#dadaebFF', '#bcbddcF0', '#9e9ac8F0',
     '#807dbaF0', '#6a51a3F0', '#54278fF0']; colors = 'Purples';

    elif color == 2: color_sq = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494']; colors = 'YlGnBu';
    elif color == 3: color_sq = ['#f7f7f7', '#d9d9d9', '#bdbdbd', '#969696', '#636363', '#252525'];  colors = 'Greys';
    elif color == 9: color_sq = ['#ff0000', '#ff0000', '#ff0000',  '#ff0000', '#ff0000', '#ff0000']
    else:
     color_sq = ['#ffffd4', '#fee391', '#fec44f', '#fe9929', '#d95f0e', '#993404']; colors = 'YlOrBr';
    new_data, bins = pd.qcut(data, 6, retbins=True,
                             labels=list(range(6)))
    color_ton = []
    for val in new_data:
        color_ton.append(color_sq[val])
    if color != 9:
        colors = sns.color_palette(colors, n_colors=6)
        sns.palplot(colors, 0.6);
        for i in range(6):
            print("\n" + str(i + 1) + ': ' + str(int(bins[i])) +
                  " => " + str(int(bins[i + 1]) - 1), end=" ")
        print("\n\n   1   2   3   4   5   6")
    return color_ton, bins;



def plot_comunas_data(sf, title, comunas, data=None,
                      color=None, print_id=False):
    '''
    Plot map with selected comunes, using specific color
    '''

    color_ton, bins = calc_color(data, color)
    print(df.head())
    comuna_id = range(54)
    plot_map_fill_multiples_ids_tone(sf, title, comuna_id,
                                     print_id,
                                     color_ton,
                                     bins,
                                     x_lim=None,
                                     y_lim=None,
                                     figsize=(15, 10));

def plot_map_fill_multiples_ids_tone(sf, title, comuna,
                                     print_id, color_ton,
                                     bins,
                                     x_lim=None,
                                     y_lim=None,
                                     figsize=(11, 9)):
    '''
    Plot map with lim coordinates
    '''

    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)


    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x, y, 'k')

    for id in comuna:
        shape_ex = sf.shape(id)
        x_lon = np.zeros((len(shape_ex.points), 1))
        y_lat = np.zeros((len(shape_ex.points), 1))
        for ip in range(len(shape_ex.points)):
            x_lon[ip] = shape_ex.points[ip][0]
            y_lat[ip] = shape_ex.points[ip][1]
        ax.fill(x_lon, y_lat, color_ton[comuna.index(id)])
        if print_id != False:
            x0 = np.mean(x_lon)
            y0 = np.mean(y_lat)
            plt.text(x0, y0, id, fontsize=1)
    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

print_id = True # The shape id will be printed
color_pallete = 4 # 'Purples'
plot_comunas_data(sf, 'Singapore Population Density',combined['street'].tolist(), data=housing_df['total_dwelling_units'].drop_duplicates().tolist(), color=color_pallete, print_id=print_id)
plt.show()



plt.show()



