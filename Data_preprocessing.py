# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
# from green_driver_trend_contribution import *

version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import re

from osgeo import ogr
from osgeo import osr
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random
# import h5py
from netCDF4 import Dataset
import shutil
import requests
from lytools import *
from osgeo import gdal

from osgeo import gdal

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
from pprint import pprint
T=Tools()
D = DIC_and_TIF(pixelsize=0.5)

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)



this_root = 'E:\Project4\\'
data_root = 'E:/Project4/Data/'
result_root = 'E:/Project4/Result/'

class Data_preprocessing:
    def __init__(self):
        pass
    def run(self):
        # self.aggregate()
        self.resample_landcover()
        # self.calculate_long_term_max()
        # self.tif_to_dic()
        # self.extract_GS()
        # self.extend_nan()
        # self.relative_change()
        # self.zscore()
        # self.trend_analysis()
        # self.detrend()
        # self.coefficient_of_variation()


        pass

    def aggregate(self):
        fdir_all= data_root + rf'biweekly_resample_NDVI4g\\'

        outdir = rf'D:\Project4\Data\\monthly_NDVI4g\\'
        Tools().mk_dir(outdir, force=True)

        year_list = list(range(1982, 2021))
        month_list = list(range(1, 13))

        for year in tqdm(year_list):
            for month in tqdm(month_list):

                data_list = []
                for f in tqdm(os.listdir(fdir_all)):
                    if not f.endswith('.tif'):
                        continue

                    data_year = f.split('.')[0][0:4]
                    data_month = f.split('.')[0][4:6]

                    if not int(data_year) == year:
                        continue
                    if not int(data_month) == int(month):
                        continue
                    arr = ToRaster().raster2array(fdir_all + f)[0]
                    # arr=arr/1000 ###
                    arr_unify = arr[:720][:720,
                                :1440]
                    arr_unify = np.array(arr_unify)
                    arr_unify=arr_unify/10000
                    arr_unify[arr_unify == 65535] = np.nan
                    arr_unify[arr_unify < 0] = np.nan
                    arr_unify[arr_unify > 1] = np.nan
                    # 当变量是LAI 的时候，<0!!
                    data_list.append(arr_unify)
                data_list = np.array(data_list)
                print(data_list.shape)
                # print(len(data_list))
                # exit()

                ##define arr_average and calculate arr_average

                arr_average = np.nanmax(data_list, axis=0)
                arr_average = np.array(arr_average)
                arr_average[arr_average < 0] = np.nan
                arr_average[arr_average > 1] = np.nan
                if np.isnan(np.nanmean(arr_average)):
                    continue
                if np.nanmean(arr_average) < 0.:
                    continue
                # plt.imshow(arr_average)
                # plt.title(f'{year}{month}')
                # plt.show()

                # save

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_average, outdir + '{}{:02d}.tif'.format(year, month))
        pass
    def resample_landcover(self):


        fdir = rf'D:\Project4\Data\Base_data\IPCC_climate_zone_v3\\'
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue

            outdir=rf'D:\Project4\Data\Base_data\IPCC_climate_zone_v3\\'
            Tools().mk_dir(outdir,force=True)

            outf = outdir + f'{f.split(".")[0]}_resample.tif'

            dataset = gdal.Open(fdir + f)

            try:
                gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass

    def calculate_long_term_max(self):
        fdir_all = data_root + rf'monthly_NDVI4g\\monthly_NDVI4g\\'
        outdir = data_root + rf'monthly_NDVI4g\\annual_max_NDVI4g\\'
        Tools().mk_dir(outdir, force=True)

        data_list = []
        for f in os.listdir(fdir_all):
            if not f.endswith('.tif'):
                continue
            arr = ToRaster().raster2array(fdir_all + f)[0]
            arr_unify = arr[:720][:720,
                        :1440]
            arr_unify = np.array(arr_unify)
            arr_unify[arr_unify == 65535] = np.nan
            arr_unify[arr_unify < 0] = np.nan
            arr_unify[arr_unify > 1] = np.nan
            data_list.append(arr_unify)
        data_list = np.array(data_list)
        arr_average = np.nanmax(data_list, axis=0)
        arr_average = np.array(arr_average)
        arr_average[arr_average < 0] = np.nan
        arr_average[arr_average > 1] = np.nan
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_average, outdir + '1982_2020_max.tif')

        pass


    def tif_to_dic(self):
        fdir_all = data_root+rf'\monthly_LAI4g\\'

        NDVI_mask_f = data_root + rf'Base_data\annual_max_NDVI4g\\1982_2020_max.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0.2] = np.nan

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'monthly_LAI4g' in fdir:
                continue

            outdir = data_root + rf'Monthly_DIC\\LAI4g\\'
            # if os.path.isdir(outdir):
            #     pass

            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all + fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir_all + fdir + '\\' + f)
                array = np.array(array, dtype=float)

                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify < -999] = np.nan

                array_unify=array_unify

                array_unify[array_unify < 0] = np.nan

                # plt.imshow(array)
                # plt.show()
                array_mask = np.array(array_mask, dtype=float)
                # plt.imshow(array_mask)
                # plt.show()
                array_dryland = array_unify * array_mask
                # plt.imshow(array_dryland)
                # plt.show()

                all_array.append(array_dryland)

            row = len(all_array[0])
            col = len(all_array[0][0])
            key_list = []
            dic = {}

            for r in tqdm(range(row), desc='构造key'):  # 构造字典的键值，并且字典的键：值初始化
                for c in range(col):
                    dic[(r, c)] = []
                    key_list.append((r, c))
            # print(dic_key_list)

            for r in tqdm(range(row), desc='构造time series'):  # 构造time series
                for c in range(col):
                    for arr in all_array:
                        value = arr[r][c]
                        dic[(r, c)].append(value)
                    # print(dic)
            time_series = []
            flag = 0
            temp_dic = {}
            for key in tqdm(key_list, desc='output...'):  # 存数据
                flag = flag + 1
                time_series = dic[key]
                time_series = np.array(time_series)
                temp_dic[key] = time_series
                if flag % 10000 == 0:
                    # print(flag)
                    np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)


    def extract_GS(self):  ## here using new extraction method: 240<r<480 all year growing season

        fdir_all = data_root + rf'\Monthly_DIC\\'
        outdir = result_root + f'extract_GS\\OBS_LAI\\'
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        # print(date_list)
        # exit()

        for year in range(1982, 2021):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))

        for fdir in os.listdir(fdir_all):
            if  not 'CRU' in fdir:
                continue


            spatial_dict = {}
            outf = outdir + fdir + '.npy'
            # print(outf)
            # exit()

            # if os.path.isfile(outf):
            #     continue
            print(outf)


            for f in os.listdir(fdir_all + fdir):

                spatial_dict_i =dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())
                spatial_dict.update(spatial_dict_i)

            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                r,c=pix


                gs_mon = self.global_get_gs(pix)

                vals = spatial_dict[pix]
                vals = np.array(vals)


                vals[vals < -999] = np.nan

                if T.is_all_nan(vals):
                    continue

                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)

                        date_list_index.append(i)

                consecutive_ranges = self.group_consecutive_vals(date_list_index)
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []

                if len(consecutive_ranges[0])>12:
                    consecutive_ranges=np.reshape(consecutive_ranges,(-1,12))

                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year

                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    # total_amount= np.nansum(vals_gs)  ### 降雨需要求和

                    annual_gs_list.append(mean)
                    # annual_gs_list.append(total_amount)

                annual_gs_list = np.array(annual_gs_list)

                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            np.save(outf, annual_spatial_dict)

        pass

    def global_get_gs(self,pix):
        global_northern_hemi_gs = (5, 6, 7, 8, 9, 10)
        global_southern_hemi_gs = (11, 12, 1, 2, 3, 4)
        tropical_gs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        r, c = pix
        ### 23.5°N and 23.5°S
        if r < 134:
            return global_northern_hemi_gs
        elif 134 <= r < 227:
            return tropical_gs
        elif r >= 227:
            return global_southern_hemi_gs
        else:
            raise ValueError('r not in range')

    def group_consecutive_vals(self, in_list):
        # 连续值分组
        ranges = []
        #### groupby 用法
        ### when in_list=468, how to groupby
        for _, group in groupby(enumerate(in_list), lambda index_item: index_item[0] - index_item[1]):

            group = list(map(itemgetter(1), group))
            # print(group)
            # exit()
            if len(group) > 1:
                ranges.append(list(range(group[0], group[-1] + 1)))
            else:
                ranges.append([group[0]])
        return ranges

    def extend_nan(self):
        fdir= result_root + rf'\extract_GS\OBS_LAI\\'
        outdir=result_root + rf'extract_GS\OBS_LAI_extend\\'
        T.mk_dir(outdir,force=True)
        for f in os.listdir(fdir):

            if not f.endswith('.npy'):
                continue


            outf=outdir+f.split('.')[0]+'.npy'
            # if os.path.isfile(outf):
            #     continue
            dic = dict(np.load(fdir +f, allow_pickle=True, ).item())
            dic_new={}

            for pix in tqdm(dic):
                r,c=pix

                time_series=dic[pix]

                time_series=np.array(time_series)
                time_series[time_series<-999]=np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if len(time_series)<39:

                    time_series_new=np.append(time_series,np.nan)
                    dic_new[pix]=time_series_new
                else:
                    dic_new[pix]=time_series
            np.save(outf,dic_new)

    def relative_change(self):


        fdir = result_root+'extract_GS\OBS_LAI_extend\\\\'
        outdir=result_root + rf'relative_change\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue


            outf=outdir+f.split('.')[0]+'.npy'
            if isfile(outf):
                continue
            print(outf)


            dic=T.load_npy(fdir+f)

            zscore_dic = {}

            for pix in tqdm(dic):
                delta_time_series_list = []

                # print(len(dic[pix]))
                time_series = dic[pix]
                # print(len(time_series))

                time_series = np.array(time_series)
                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series=time_series
                mean=np.nanmean(time_series)
                relative_change=(time_series-mean)/abs(mean) *100

                zscore_dic[pix] = relative_change
                ## plot
                # plt.plot(time_series)
                # plt.show()
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                plt.show()

                ## save
            np.save(outf, zscore_dic)


    def zscore(self):


        fdir = result_root+'extract_GS\OBS_LAI_extend\\\\'
        outdir=result_root + rf'zscore\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf=outdir+f.split('.')[0]+'.npy'
            if isfile(outf):
                continue
            print(outf)


            dic=T.load_npy(fdir+f)

            zscore_dic = {}

            for pix in tqdm(dic):
                delta_time_series_list = []

                # print(len(dic[pix]))
                time_series = dic[pix]
                # print(len(time_series))

                time_series = np.array(time_series)
                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series=time_series
                mean=np.nanmean(time_series)
                std=np.nanstd(time_series)
                relative_change=(time_series-mean)/std

                zscore_dic[pix] = relative_change
                ## plot
                # plt.plot(time_series)
                # plt.show()
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)




    def trend_analysis(self):

        # landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        # crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        # MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        # dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = result_root+rf'\relative_change\\'
        outdir = result_root + rf'trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue


            outf=outdir+f.split('.')[0]
            if os.path.isfile(outf+'_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r,c=pix
                # if r<120:
                #     continue
                # landcover_value=crop_mask[pix]
                # if landcover_value==16 or landcover_value==17 or landcover_value==18:
                #     continue
                # if dic_modis_mask[pix]==12:
                #     continue

                time_series=dic[pix]
                # print(time_series)


                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue

            arr_trend = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)

            p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(p_value_dic)



            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend, outf + '_trend.tif')
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    def detrend(self):

        fdir=result_root + rf'\extract_GS\OBS_LAI_extend\\'
        outdir=result_root + rf'Detrend\detrend_raw\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            print(f)

            outf=outdir+f.split('.')[0]+'.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

            detrend_zscore_dic={}

            for pix in tqdm(dic):

                r, c= pix


                # print(len(dic[pix]))
                time_series = dic[pix]
                # print(len(time_series))
                # print(time_series)

                time_series=np.array(time_series)
                # plt.plot(time_series)
                # plt.show()

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series))/len(time_series) > 0.5:
                    continue


                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                # if np.isnan(time_series).any():
                #     continue
                if r<454:
                    # print(time_series)
                    ### interpolate
                    time_series=T.interp_nan(time_series)
                    # print(np.nanmean(time_series))
                    # plt.plot(time_series)



                    detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                    # plt.plot(detrend_delta_time_series)
                    # plt.show()


                    detrend_zscore_dic[pix] = detrend_delta_time_series
                else:
                    time_series=time_series[0:38]
                    print(time_series)


                    if np.isnan(time_series).any():
                        continue
                    # print(time_series)
                    detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                    ###add nan to the end if length is less than time_series
                    if len(detrend_delta_time_series) < 39:
                        detrend_delta_time_series=np.append(detrend_delta_time_series, [np.nan]*(39-len(detrend_delta_time_series)))

                        detrend_zscore_dic[pix] = detrend_delta_time_series


            np.save(outf, detrend_zscore_dic)

    def coefficient_of_variation(self):
        fdir = result_root + rf'Detrend\detrend_raw\\'
        outdir = result_root + rf'coefficient_of_variation\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            print(f)

            outf = outdir + f.split('.')[0] + '.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

            cv_dic = {}

            for pix in tqdm(dic):

                r, c = pix

                time_series = dic[pix]

                time_series = np.array(time_series)
                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series)) / len(time_series) > 0.5:
                    continue

                cv = np.nanstd(time_series) / np.nanmean(time_series)*100

                cv_dic[pix] = cv
            array_CV = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(cv_dic)
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_CV, outf + '.tif')

            np.save(outf, cv_dic)


class Monte_carlo_trend_extreme:  ##to analyze the trend of the extreme dry year and wet year
    def __init__(self):
        pass

    def run(self):
        # self.cal_monte_carlo_wet()
        # self.cal_monte_carlo_dry()
        # self.cal_monte_carlo_heat()
        # self.cal_monte_carlo_cold()
        # self.cal_monte_carlo_cold_compound()
        # self.cal_monte_carlo_heat_compound()
        # self.cal_monte_carlo_all_extreme()
        # self.check_result()
        # self.difference()
        # self.spatial_map()
        self.plot_probability_density_spatial()


        pass

    def cal_monte_carlo_wet(self):
        outdir = join(result_root, 'monte_carlo')
        T.mk_dir(outdir)
        mode='wet'

        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf=join(outdir,f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]

            picked_wet_index = precipitation>=1

            selected_year = year_list[picked_wet_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std



    def cal_monte_carlo_dry(self):
        mode='dry'

        outdir = join(result_root, 'monte_carlo')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_preciptation = result_root + rf'zscore\\CRU.npy'
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]
            ##percentile<10 is the extreme dry year`


            picked_dry_index = precipitation <= -1

            selected_year = year_list[picked_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat(self):
        mode='heat'

        outdir = join(result_root, 'monte_carlo')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmax = result_root + rf'zscore\\tmax.npy'

        dic_tmax = T.load_npy(f_tmax)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]


            picked_heat_index = temp >= 1

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold(self):
        mode='cold'

        outdir = join(result_root, 'monte_carlo')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmin = result_root + rf'zscore\\tmin.npy'

        dic_tmin = T.load_npy(f_tmin)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            ##percentile<10 is the extreme dry year`

            picked_heat_index = temp <= -1

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold_compound(self):
        # mode='cold_wet'
        mode='cold_dry'

        outdir = join(result_root, 'monte_carlo_trend_extreme')
        T.mk_dir(outdir)


        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmin = T.load_npy(f_tmin)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            precipitation = dic_precip[pix]

            ##selected cold and wet year/ cold and dry year


            # picked_cold_wet_index = (temp <= -1) & (precipitation >= 1)
            picked_cold_dry_index = (temp <= -1) & (precipitation <= -1)


            selected_year = year_list[picked_cold_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat_compound(self):
        # mode = 'heat_wet'
        mode='heat_dry'

        outdir = join(result_root, 'monte_carlo_trend_extreme')
        T.mk_dir(outdir)

        f_tmax = result_root + rf'zscore\\tmax.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmax = T.load_npy(f_tmax)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]
            precipitation = dic_precip[pix]
            ##selected heat wet and heat dry year


            # picked_heat_wet_index = (temp >= 1) & (precipitation >= 1)
            picked_heat_dry_index = (temp >= 1) & (precipitation <= -1)

            selected_year = year_list[picked_heat_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def kernel_cal_monte_carlo_trend(self, params):
        n = 100
        pix, lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        slope_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)

            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            a, b, r, p = T.nan_line_fit(range(len(lai_copy)), lai_copy)
            slope_list.append(a)

        mean = np.nanmean(slope_list)
        std = np.nanstd(slope_list)
        return (pix, mean, std)

        pass

    def kernel_cal_monte_carlo_CV(self, params):
        n = 100
        pix, lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        CV_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)

            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            CV=np.nanstd(lai_copy)/np.nanmean(lai_copy)*100
            CV_list.append(CV)

        mean = np.nanmean(CV_list)
        std = np.nanstd(CV_list)
        return (pix, mean, std)

        pass
    def cal_monte_carlo_all_extreme(self):  ## all extreme year events

        mode = 'all_extreme'

        outdir = join(result_root, 'monte_carlo_trend_extreme')
        T.mk_dir(outdir)



        f_tmax = result_root + rf'zscore\\tmax.npy'
        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmax = T.load_npy(f_tmax)
        dic_tmin = T.load_npy(f_tmin)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'E:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            tmax = dic_tmax[pix]
            tmin = dic_tmin[pix]
            precipitation = dic_precip[pix]
            ##extreme wet/dry and heat/cold year, and compound events all belongs to extreme year


            picked_all_extreme_index = (tmax >= 1) | (tmin <= -1) | (precipitation >= 1) | (precipitation <= -1)



            selected_year = year_list[picked_all_extreme_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

        pass
    def check_result(self):


        fdir = rf'E:\Project4\Result\monte_carlo_trend_extreme\\'
        for f in os.listdir(fdir):
            if not 'npy' in f:
                continue
            if 'slope' in f:
                continue
            if 'std' in f:
                continue
            if not 'all_extreme' in f:
                continue

            fpath = join(fdir, f)
            result_dict = np.load(fpath, allow_pickle=True)
            spatial_dict1 = {}
            spatial_dict2 = {}
            # for i in result_dict:
            #     print(i)
            #     exit()
            for pix, slope, std in result_dict:
                # if np.isnan(slope):
                #     continue
                # print(pix,slope,std)

                spatial_dict1[pix] = slope
                spatial_dict2[pix] = std
            arr1 = D.pix_dic_to_spatial_arr(spatial_dict1)
            arr2 = D.pix_dic_to_spatial_arr(spatial_dict2)
            # plt.imshow(arr1)
            # plt.colorbar()
            # plt.show()
            ##save
            D.arr_to_tif(arr1, fpath.replace('.npy', '_slope.tif'))
            D.arr_to_tif(arr2, fpath.replace('.npy', '_std.tif'))
            np.save(fpath.replace('.npy', '_slope.npy'), spatial_dict1)
            np.save(fpath.replace('.npy', '_std.npy'), spatial_dict2)

    def difference(self): ### calculate the difference of the slope the real and scenario


        f_real_trend = result_root+rf'\trend_analysis\\LAI4g_trend.tif'
        array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real_trend)
        fdir=rf'E:\Project4\Result\monte_carlo_trend_extreme\\'
        for f in os.listdir(fdir):
            if not 'slope' in f:
                continue
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            if not 'all_extreme' in f:
                continue
            fpath=join(fdir,f)
            # print(fpath)
            array_scenario, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real[array_real ==0] = np.nan
            array_scenario[array_scenario == 0] = np.nan
            array_diff = array_real - array_scenario
            array_diff[array_diff ==0] = np.nan

            fname_1=f.replace('slope','trend')
            fname=fname_1.replace('monte_carlo_','')

            outf=join(fdir,'difference', fname)
            T.mk_dir(join(fdir,'difference'))
            dic_arr_difference = D.spatial_arr_to_dic(array_diff)
            spatial_dict = {}
            for pix in tqdm(dic_arr_difference):
                r,c = pix


                # if not pix in dic_modis_mask:
                #     continue
                # if dic_modis_mask[pix] == 12:
                #     continue
                # landcover_value = crop_mask[pix]
                # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                #     continue
                spatial_dict[pix] = dic_arr_difference[pix]
            array_diff_new = D.pix_dic_to_spatial_arr(spatial_dict)
            # array_diff_new = array_diff_new * array_mask
            array_diff_new[array_diff_new > 99] = np.nan
            array_diff_new[array_diff_new < -99] = np.nan


            D.arr_to_tif(array_diff_new, outf)



    def contribution(self):
        fdir=rf'D:\Project4\Result\monte_carlo_CV\\difference\\'
        f_real=rf'D:\Project4\Result\coefficient_of_variation\\LAI4g.npy.tif'
        outdir=result_root+rf'monte_carlo_CV\\contribution\\'
        T.mk_dir(outdir,force=1)
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            array_diff, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real)
            # array_diff[array_diff == 0] = np.nan
            # array_real[array_real == 0] = np.nan
            array_diff = np.abs(array_diff)
            array_real = np.abs(array_real)
            array_contribution = array_diff/array_real*100
            array_contribution[array_contribution==100]=np.nan

            outf=join(outdir,f.replace('diff','contribution'))
            D.arr_to_tif(array_contribution, outf)


        pass

    def spatial_map(self):  ## plot the spatial map of the CV for each pft
        fdir =result_root+ rf'monte_carlo_trend_extreme\\difference\\'
        outdir = result_root + rf'monte_carlo_trend_extreme\\spatial_map_LUCC_mask\\'
        f_lucc = data_root + rf'\Base_data\MODIS_IGBP_WGS84\IGBP_reclass\IGBP_reclass_resample_rename\\2010_01_01_resample.tif.npy'
        dic_lucc = T.load_npy(f_lucc)

        T.mk_dir(outdir, force=1)
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue

            fpath = join(fdir, f)
            array_diff, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_diff[array_diff == 0] = np.nan

            array_diff[array_diff > 99] = np.nan
            array_diff[array_diff < -99] = np.nan
            ## values<0.0001 and values>-0.0001 mask to nan

            array_diff[(array_diff < 0.001) & (array_diff > -0.001)] = np.nan


            spatial_dict = DIC_and_TIF().spatial_arr_to_dic(array_diff)
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            flag = 0

            MODIS_LUCC_list = ['Forests', 'Shrubland', 'Grassland', 'Cropland']
            for LUCC in MODIS_LUCC_list:
                spatial_dict_mask = {}

                ax = axs.ravel()[flag]
                for pix in spatial_dict:
                    r, c = pix
                    if not pix in dic_lucc:
                        continue
                    lucc = dic_lucc[pix]
                    if lucc == LUCC:
                        spatial_dict_mask[pix] = spatial_dict[pix]
                    else:
                        spatial_dict_mask[pix] = np.nan
                array_diff_new = D.pix_dic_to_spatial_arr(spatial_dict_mask)
                ax.imshow(array_diff_new, cmap='jet', vmin=-0.1, vmax=0.1)
                ## add the colorbar
                cbar = plt.colorbar(ax.imshow(array_diff_new, cmap='RdBu', vmin=-0.1, vmax=0.1,interpolation='nearest'), ax=ax, orientation='horizontal')
                land_tif = join(data_root,'Base_data','land.tif')
                DIC_and_TIF().plot_back_ground_arr(land_tif, ax,interpolation='nearest')
                ax.set_title(LUCC)
                # ax.axis('off')
                flag += 1
            # plt.tight_layout()
            plt.suptitle(f.split('.')[0])

            plt.show()

                # outf = join(outdir, f.replace('.tif', f'_{LUCC}.tif'))
                # D.arr_to_tif(array_diff_new, outf)

        pass

    def plot_probability_density_spatial(self):
        df=result_root+rf'monte_carlo_trend_extreme\Dataframe\monte_carlo_trend.df'
        df=T.load_df(df)
        ## seaborn plot

        # MODIS_LUCC_list = ['Forests', 'Shrubland', 'Grassland', 'Cropland', 'ALL']
        MODIS_LUCC_list = ['Forests', 'Shrubland', 'Grassland', 'ALL']

        event_list = ['all_extreme_year_LAI','LAI4g','cold_year_LAI', 'wet_year_LAI', 'cold_dry_year_LAI', 'cold_wet_year_LAI', 'dry_year_LAI',
                      'heat_year_LAI', 'heat_dry_year_LAI', 'heat_wet_year_LAI',]



        for event in event_list:

            df_event=df.dropna(subset=['LAI4g_trend'])
            if len(df_event)==0:
                continue

            fig, axs = plt.subplots(2, 2, figsize=(10, 5))
            flag = 0

            for LUCC in MODIS_LUCC_list:
                ax = axs.ravel()[flag]
                if LUCC == 'ALL':
                    ax = axs.ravel()[flag]
                    df_sub = df_event
                else:

                    df_sub=df_event[df_event['MODIS_LUCC']==LUCC]
                values=df_sub[event+'_trend'].to_numpy()
                values = np.array(values)
                print(len(values))
                # values[values>1]=np.nan
                # values[values<-1]=np.nan
                ## remove below 0.0001 and above -0.0001
                # values=values[(values>=-0.0001) | (values<=0.0001)]
                mask_array1 = values>-0.001
                # print(mask_array1)
                mask_array2 = values<0.001
                mask_array = np.logical_and(mask_array1, mask_array2)
                print(mask_array)
                values[mask_array] = np.nan

                # values[values>-0.0001]=np.nan


                values = values[~np.isnan(values)]

                print(len(values))

                # if len(values) ==0:
                #     continue
                ## plot bimodel
                Plot().plot_hist_smooth(values,range=(-0.3,0.3),bins=1000,ax=ax)
                # plt.show()
                # weights = np.ones_like(values) / float(len(values))
                # ax.hist(values, bins=1000, density=True, alpha=0.6, color='g',weights=weights,range=(-0.5,0.5))
                # counts, bins = np.histogram(values, bins=1000, density=True,range=(-0.5,0.5))
                # print(len(counts),len(values))
                # ax.hist(bins[:-1], bins, weights=counts, alpha=0.6, color='g')
                # sns.distplot(values, hist=False,bins=np.arange(-0.5,0.5,0.01),ax=ax)
                # print();exit()
                # ax.hist(values, bins=1000, density=True, alpha=0.6, color='g',range=(-0.5,0.5),weights=counts)
                # sns.kdeplot(values, color='k',ax=ax)
                ## set line x=0
                # ax.axvline(0, color='grey', linewidth=1)
                ax.set_title(LUCC)
                ax.set_xlim(-0.5, 0.5)
                # ax.set_ylim(0, 5)
                ax.set_title(LUCC)
                ax.set_xlabel('Trend %')

                flag += 1
            plt.suptitle(event)
            plt.show()



class Monte_carlo_trend_moderate:
    def __init__(self):
        pass

    def run(self):
        # self.cal_monte_carlo_wet()
        # self.cal_monte_carlo_dry()
        # self.cal_monte_carlo_heat()
        # self.cal_monte_carlo_cold()
        # self.cal_monte_carlo_cold_compound()
        # self.cal_monte_carlo_heat_compound()
        self.check_result()
        self.difference()


        pass

    def cal_monte_carlo_wet(self):
        outdir = join(result_root, 'monte_carlo_trend_moderate')
        T.mk_dir(outdir)
        mode='wet'

        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf=join(outdir,f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]

            picked_wet_index = (precipitation>=0.5) & (precipitation<1)

            selected_year = year_list[picked_wet_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std



    def cal_monte_carlo_dry(self):
        mode='dry'

        outdir = join(result_root, 'monte_carlo_trend_moderate')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_preciptation = result_root + rf'zscore\\CRU.npy'
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]
            ##percentile<10 is the extreme dry year`


            picked_dry_index = (precipitation <= -0.5) & (precipitation > -1)

            selected_year = year_list[picked_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat(self):
        mode='heat'

        outdir = join(result_root, 'monte_carlo_trend_moderate')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmax = result_root + rf'zscore\\tmax.npy'

        dic_tmax = T.load_npy(f_tmax)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]


            picked_heat_index = (temp >= 0.5) & (temp < 1)

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold(self):
        mode='cold'

        outdir = join(result_root, 'monte_carlo_trend_moderate')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmin = result_root + rf'zscore\\tmin.npy'

        dic_tmin = T.load_npy(f_tmin)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            ##percentile<10 is the extreme dry year`

            picked_heat_index = (temp <= -0.5) & (temp > -1)

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold_compound(self):
        # mode='cold_wet'
        mode='cold_dry'

        outdir = join(result_root, 'monte_carlo_trend_moderate')
        T.mk_dir(outdir)


        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmin = T.load_npy(f_tmin)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            precipitation = dic_precip[pix]

            ##selected cold and wet year/ cold and dry year


            # picked_cold_wet_index = (temp <= -0.5) & (temp > -1) & (precipitation >= 0.5) & (precipitation < 1)
            picked_cold_dry_index = (temp <= -0.5) & (temp > -1) & (precipitation <= -0.5) & (precipitation > -1)


            selected_year = year_list[picked_cold_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat_compound(self):
        # mode = 'heat_wet'
        mode='heat_dry'

        outdir = join(result_root, 'monte_carlo_trend_moderate')
        T.mk_dir(outdir)

        f_tmax = result_root + rf'zscore\\tmax.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmax = T.load_npy(f_tmax)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\relative_change\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')


        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]
            precipitation = dic_precip[pix]
            ##selected heat wet and heat dry year


            # picked_heat_wet_index = (temp >= 0.5) & (temp < 1) & (precipitation >= 0.5) & (precipitation < 1)
            picked_heat_dry_index = (temp >= 0.5) & (temp < 1) & (precipitation <= -0.5) & (precipitation > -1)

            selected_year = year_list[picked_heat_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_trend, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def kernel_cal_monte_carlo_trend(self, params):
        n = 100
        pix, lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        slope_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)

            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            a, b, r, p = T.nan_line_fit(range(len(lai_copy)), lai_copy)
            slope_list.append(a)

        mean = np.nanmean(slope_list)
        std = np.nanstd(slope_list)
        return (pix, mean, std)

        pass

    def kernel_cal_monte_carlo_CV(self, params):
        n = 100
        pix, lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        CV_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)

            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            CV=np.nanstd(lai_copy)/np.nanmean(lai_copy)*100
            CV_list.append(CV)

        mean = np.nanmean(CV_list)
        std = np.nanstd(CV_list)
        return (pix, mean, std)

        pass

    def check_result(self):


        fdir = rf'D:\Project4\Result\monte_carlo_trend_moderate\\'
        for f in os.listdir(fdir):
            if not 'npy' in f:
                continue
            if 'slope' in f:
                continue
            if 'std' in f:
                continue

            fpath = join(fdir, f)
            result_dict = np.load(fpath, allow_pickle=True)
            spatial_dict1 = {}
            spatial_dict2 = {}
            # for i in result_dict:
            #     print(i)
            #     exit()
            for pix, slope, std in result_dict:
                # if np.isnan(slope):
                #     continue
                # print(pix,slope,std)

                spatial_dict1[pix] = slope
                spatial_dict2[pix] = std
            arr1 = D.pix_dic_to_spatial_arr(spatial_dict1)
            arr2 = D.pix_dic_to_spatial_arr(spatial_dict2)
            # plt.imshow(arr1)
            # plt.colorbar()
            # plt.show()
            ##save
            D.arr_to_tif(arr1, fpath.replace('.npy', '_slope.tif'))
            D.arr_to_tif(arr2, fpath.replace('.npy', '_std.tif'))
            np.save(fpath.replace('.npy', '_slope.npy'), spatial_dict1)
            np.save(fpath.replace('.npy', '_std.npy'), spatial_dict2)

    def difference(self): ### calculate the difference of the slope the real and scenario


        f_real_trend = result_root+rf'\trend_analysis\\LAI4g_trend.tif'
        array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real_trend)
        fdir=rf'D:\Project4\Result\monte_carlo_trend_moderate\\'
        for f in os.listdir(fdir):
            if not 'slope' in f:
                continue
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            # print(fpath)
            array_scenario, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real[array_real ==0] = np.nan
            array_scenario[array_scenario == 0] = np.nan
            array_diff = array_real - array_scenario
            array_diff[array_diff ==0] = np.nan

            fname_1=f.replace('slope','trend')
            fname=fname_1.replace('monte_carlo_','')

            outf=join(fdir,'difference', fname)
            T.mk_dir(join(fdir,'difference'))
            dic_arr_difference = D.spatial_arr_to_dic(array_diff)
            spatial_dict = {}
            for pix in tqdm(dic_arr_difference):
                r,c = pix


                # if not pix in dic_modis_mask:
                #     continue
                # if dic_modis_mask[pix] == 12:
                #     continue
                # landcover_value = crop_mask[pix]
                # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                #     continue
                spatial_dict[pix] = dic_arr_difference[pix]
            array_diff_new = D.pix_dic_to_spatial_arr(spatial_dict)
            # array_diff_new = array_diff_new * array_mask
            array_diff_new[array_diff_new > 99] = np.nan
            array_diff_new[array_diff_new < -99] = np.nan


            D.arr_to_tif(array_diff_new, outf)



    def contribution(self):
        fdir=rf'D:\Project4\Result\monte_carlo_CV\\difference\\'
        f_real=rf'D:\Project4\Result\coefficient_of_variation\\LAI4g.npy.tif'
        outdir=result_root+rf'monte_carlo_CV\\contribution\\'
        T.mk_dir(outdir,force=1)
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            array_diff, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real)
            # array_diff[array_diff == 0] = np.nan
            # array_real[array_real == 0] = np.nan
            array_diff = np.abs(array_diff)
            array_real = np.abs(array_real)
            array_contribution = array_diff/array_real*100
            array_contribution[array_contribution==100]=np.nan

            outf=join(outdir,f.replace('diff','contribution'))
            D.arr_to_tif(array_contribution, outf)


        pass


class Monte_carlo_CV_extreme:
    def __init__(self):
        pass
    def run(self):
        # self.cal_monte_carlo_wet()
        # self.cal_monte_carlo_dry()
        # self.cal_monte_carlo_heat()
        # self.cal_monte_carlo_cold()
        # self.cal_monte_carlo_heat_compound()
        # self.cal_monte_carlo_cold_compound()
        # self.cal_monte_carlo_all_extreme()
        # self.check_result()
        # self.difference()
        # self.contribution()
        # self.spatial_map()
        self.plot_probability_density_spatial()

        pass
    pass

    def cal_monte_carlo_wet(self):
        outdir = join(result_root, 'monte_carlo_CV')
        T.mk_dir(outdir)
        mode = 'wet'

        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]

            picked_wet_index = precipitation >= 1

            selected_year = year_list[picked_wet_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_dry(self):
        mode = 'dry'

        outdir = join(result_root, 'monte_carlo_CV')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_preciptation = result_root + rf'zscore\\CRU.npy'
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]
            ##percentile<10 is the extreme dry year`

            picked_dry_index = precipitation <= -1

            selected_year = year_list[picked_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat(self):
        mode = 'heat'

        outdir = join(result_root, 'monte_carlo_CV')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmax = result_root + rf'zscore\\tmax.npy'

        dic_tmax = T.load_npy(f_tmax)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]

            picked_heat_index = temp >= 1

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold(self):
        mode = 'cold'

        outdir = join(result_root, 'monte_carlo_CV')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmin = result_root + rf'zscore\\tmin.npy'

        dic_tmin = T.load_npy(f_tmin)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            ##percentile<10 is the extreme dry year`

            picked_heat_index = temp <= -1

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold_compound(self):
        mode = 'cold_wet'
        # mode= 'cold_dry'

        outdir = join(result_root, 'monte_carlo_CV')
        T.mk_dir(outdir)

        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmin = T.load_npy(f_tmin)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            precipitation = dic_precip[pix]
            ##selected cold and wet year/ cold and dry year

            picked_cold_wet_index = (temp <= -1.5) & (precipitation >= 1.5)
            # picked_cold_dry_index = (temp <= -1) & (precipitation <= -1)

            selected_year = year_list[picked_cold_wet_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat_compound(self):
        mode = 'heat_wet'
        # mode='heat_dry'

        outdir = join(result_root, 'monte_carlo_CV')
        T.mk_dir(outdir)

        f_tmax = result_root + rf'zscore\\tmax.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmax = T.load_npy(f_tmax)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmax:
                continue
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]

            precipitation = dic_precip[pix]

            ##selected heat wet and heat dry year

            picked_heat_wet_index = (temp >= 1) & (precipitation >= 1)
            # picked_heat_dry_index = (temp >= 1) & (precipitation <= -1)

            selected_year = year_list[picked_heat_wet_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_all_extreme(self):  ## all extreme year events

        mode = 'all_extreme'

        outdir = join(result_root, 'monte_carlo_CV_extreme')
        T.mk_dir(outdir)


        f_tmax = result_root + rf'zscore\\tmax.npy'
        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmax = T.load_npy(f_tmax)
        dic_tmin = T.load_npy(f_tmin)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = result_root + rf'\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            tmax = dic_tmax[pix]
            tmin = dic_tmin[pix]
            precipitation = dic_precip[pix]
            ##extreme wet/dry and heat/cold year, and compound events all belongs to extreme year


            picked_all_extreme_index = (tmax >= 1) | (tmin <= -1) | (precipitation >= 1) | (precipitation <= -1)



            selected_year = year_list[picked_all_extreme_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std



    def kernel_cal_monte_carlo_CV(self, params):
        n = 100
        pix, lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        CV_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)

            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            CV = np.nanstd(lai_copy) / np.nanmean(lai_copy) * 100
            CV_list.append(CV)

        mean = np.nanmean(CV_list)
        std = np.nanstd(CV_list)
        return (pix, mean, std)

        pass

    def check_result(self):

        fdir = rf'E:\Project4\Result\monte_carlo_CV_extreme\\'
        for f in os.listdir(fdir):
            if not 'npy' in f:
                continue
            if 'slope' in f:
                continue
            if 'std' in f:
                continue
            if not 'all_extreme' in f:
                continue

            fpath = join(fdir, f)
            result_dict = np.load(fpath, allow_pickle=True)
            spatial_dict1 = {}
            spatial_dict2 = {}
            # for i in result_dict:
            #     print(i)
            #     exit()
            for pix, slope, std in result_dict:
                # if np.isnan(slope):
                #     continue
                # print(pix,slope,std)

                spatial_dict1[pix] = slope
                spatial_dict2[pix] = std
            arr1 = D.pix_dic_to_spatial_arr(spatial_dict1)
            arr2 = D.pix_dic_to_spatial_arr(spatial_dict2)
            # plt.imshow(arr1)
            # plt.colorbar()
            # plt.show()
            ##save
            D.arr_to_tif(arr1, fpath.replace('.npy', '_slope.tif'))
            D.arr_to_tif(arr2, fpath.replace('.npy', '_std.tif'))
            np.save(fpath.replace('.npy', '_slope.npy'), spatial_dict1)
            np.save(fpath.replace('.npy', '_std.npy'), spatial_dict2)


    def difference(self): ### calculate the difference of the slope the real and scenario
        # NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        # array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        # landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        # crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        # MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        # dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        f_real_trend = result_root+rf'coefficient_of_variation\\LAI4g.npy.tif'
        array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real_trend)
        fdir=rf'E:\Project4\Result\monte_carlo_CV_extreme\\'
        for f in os.listdir(fdir):
            if not 'slope' in f:
                continue
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            if not 'all_extreme' in f:
                continue
            fpath=join(fdir,f)
            # print(fpath)
            array_scenario, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real[array_real ==0] = np.nan
            array_scenario[array_scenario == 0] = np.nan
            array_diff = array_real - array_scenario
            array_diff[array_diff ==0] = np.nan

            fname_1=f.replace('slope','CV')
            fname=fname_1.replace('monte_carlo_','')

            outf=join(fdir,'difference', fname)
            T.mk_dir(join(fdir,'difference'))
            dic_arr_difference = D.spatial_arr_to_dic(array_diff)
            spatial_dict = {}
            for pix in tqdm(dic_arr_difference):
                r,c = pix


                # if not pix in dic_modis_mask:
                #     continue
                # if dic_modis_mask[pix] == 12:
                #     continue
                # landcover_value = crop_mask[pix]
                # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                #     continue
                spatial_dict[pix] = dic_arr_difference[pix]
            array_diff_new = D.pix_dic_to_spatial_arr(spatial_dict)
            # array_diff_new = array_diff_new * array_mask
            array_diff_new[array_diff_new > 99] = np.nan
            array_diff_new[array_diff_new < -99] = np.nan


            D.arr_to_tif(array_diff_new, outf)

    def contribution(self):
        fdir=rf'D:\Project4\Result\monte_carlo_CV_extreme\\difference\\'
        f_real=rf'D:\Project4\Result\coefficient_of_variation\\LAI4g.npy.tif'
        outdir=result_root+rf'monte_carlo_CV_extreme\\contribution\\'
        T.mk_dir(outdir,force=1)
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            array_diff, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real)
            # array_diff[array_diff == 0] = np.nan
            # array_real[array_real == 0] = np.nan
            array_diff = np.abs(array_diff)
            array_real = np.abs(array_real)
            array_contribution = array_diff/array_real*100
            array_contribution[array_contribution==100]=np.nan

            outf=join(outdir,f.replace('diff','contribution'))
            D.arr_to_tif(array_contribution, outf)


        pass
    def spatial_map(self):  ## plot the spatial map of the CV for each pft
        fdir=result_root+rf'\monte_carlo_CV_extreme\\difference\\'
        outdir=result_root+rf'monte_carlo_CV_extreme\\spatial_map_LUCC_mask\\'
        f_lucc = data_root + rf'\Base_data\MODIS_IGBP_WGS84\IGBP_reclass\IGBP_reclass_resample_rename\\2010_01_01_resample.tif.npy'
        dic_lucc = T.load_npy(f_lucc)


        T.mk_dir(outdir,force=1)
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue

            if 'xml' in f:
                continue
            if not 'all_extreme' in f:
                continue
            fpath=join(fdir,f)
            array_diff, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_diff[array_diff == 0] = np.nan
            array_diff[array_diff > 99] = np.nan
            array_diff[array_diff < -99] = np.nan
            spatial_dict=DIC_and_TIF().spatial_arr_to_dic(array_diff)

            MODIS_LUCC_list=['Forests','Shrubland','Grassland','Cropland']
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            flag = 0
            for LUCC in MODIS_LUCC_list:
                spatial_dict_mask = {}

                ax = axs.ravel()[flag]
                for pix in spatial_dict:
                    r, c = pix
                    if not pix in dic_lucc:
                        continue
                    lucc = dic_lucc[pix]
                    if lucc == LUCC:
                        spatial_dict_mask[pix] = spatial_dict[pix]
                    else:
                        spatial_dict_mask[pix] = np.nan
                array_diff_new = D.pix_dic_to_spatial_arr(spatial_dict_mask)
                ax.imshow(array_diff_new, cmap='jet', vmin=-1, vmax=1)
                ## add the colorbar
                cbar = plt.colorbar(
                    ax.imshow(array_diff_new, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest'), ax=ax,
                    orientation='horizontal')
                land_tif = join(data_root, 'Base_data', 'land.tif')
                DIC_and_TIF().plot_back_ground_arr(land_tif, ax, interpolation='nearest')
                ax.set_title(LUCC)
                # ax.axis('off')
                flag += 1
            # plt.tight_layout()
            plt.suptitle(f.split('.')[0])

            plt.show()
            # outf=join(outdir,f.replace('.tif',f'_{LUCC}.tif'))
            # D.arr_to_tif(array_diff_new, outf)



        pass

    def plot_probability_density_spatial(self):
        df=result_root+rf'monte_carlo_CV_extreme\Dataframe\monte_carlo_CV.df'
        df=T.load_df(df)
        ## seaborn plot

        # MODIS_LUCC_list = ['Forests', 'Shrubland', 'Grassland', 'Cropland', 'ALL']
        MODIS_LUCC_list = ['Forests', 'Shrubland', 'Grassland', 'ALL']

        event_list = ['all_extreme_year_LAI','LAI4g','cold_year_LAI', 'wet_year_LAI', 'cold_dry_year_LAI', 'cold_wet_year_LAI', 'dry_year_LAI',
                      'heat_year_LAI', 'heat_dry_year_LAI', 'heat_wet_year_LAI',]



        for event in event_list:

            df_event=df.dropna(subset=['LAI4g_CV'])
            if len(df_event)==0:
                continue

            fig, axs = plt.subplots(2, 2, figsize=(10, 5))
            flag = 0

            for LUCC in MODIS_LUCC_list:
                ax = axs.ravel()[flag]
                if LUCC == 'ALL':
                    ax = axs.ravel()[flag]
                    df_sub = df_event
                else:

                    df_sub=df_event[df_event['MODIS_LUCC']==LUCC]
                values=df_sub[event+'_CV'].to_numpy()
                # values[values>5]=np.nan
                # values[values<-5]=np.nan
                values=values[~np.isnan(values)]
                if len(values) ==0:
                    continue
                ## plot bimodel

                ax.hist(values, bins=100, density=True, alpha=0.6, color='g')
                sns.kdeplot(values, color='k',ax=ax)
                ## set line x=0
                ax.axvline(0, color='grey', linewidth=1)
                ax.set_title(LUCC)
                ax.set_xlim(-3,3)
                ax.set_ylim(0,3)
                ax.set_title(LUCC)
                ax.set_xlabel('CV %')

                flag += 1
            plt.suptitle(event)
            plt.show()



    def barplot(self):

        dff = rf'D:\Project3\Result\monte_carlo\Dataframe\monte_carlo.df'
        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # df=df[df['continent']=='Australia']
        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']>0]
        df = df.dropna()
        dry_color_list = ['peachpuff', 'orange', 'darkorange', 'chocolate', 'saddlebrown']

        wet_color_list = ['lightblue', 'cyan', 'deepskyblue', 'dodgerblue', 'navy']
        color_list = wet_color_list + dry_color_list[::-1]
        # df=df[df['partial_corr_GPCC']<0.4]
        df = df.dropna()
        # print(df.columns)
        # exit()
        #### plt. bar plot for the wet and dry year all the period

        result_period_dic = {}
        mode_list = ['extreme', 'moderate', 'mild', ]
        patterns = ['wet', 'dry']

        period_list = ['1982_2000', '2001_2020', '1982_2020']

        for period in period_list:

            lai_raw = df[f'LAI4g_{period}_trend'].to_list()
            lai_raw = np.array(lai_raw)
            lai_raw_mean = np.nanmean(lai_raw)
            lai_raw_std = np.nanstd(lai_raw)

            result_dic_pattern = {}
            for pattern in patterns:
                result_dic_mode = {}
                for mode in mode_list:
                    vals = df[f'{pattern}_year_trend_{mode}_{period}'].to_list()
                    vals_array = np.array(vals)
                    vals_mean = np.nanmean(vals_array)
                    vals_std = np.nanstd(vals_array)
                    result_dic_mode[mode] = (vals_mean, vals_std)
                ##add the mean of the raw data
                vals_mean = np.nanmean(lai_raw)
                vals_std = np.nanstd(lai_raw)
                result_dic_mode['raw'] = (vals_mean, vals_std)

                result_dic_pattern[pattern] = result_dic_mode
            result_period_dic[period] = result_dic_pattern

        ##plot
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        flag = 0
        for period in period_list:
            ax = axs.ravel()[flag]
            result_dic_pattern = result_period_dic[period]
            vals_mean_list = []
            vals_std_list = []
            for pattern in patterns:
                result_dic_mode = result_dic_pattern[pattern]

                for mode in mode_list:
                    vals_mean, vals_std = result_dic_mode[mode]
                    vals_mean_list.append(vals_mean)
                    vals_std_list.append(vals_std)
            mode_list_new = [f'{mode}_{pattern}' for pattern in patterns for mode in mode_list]

            ax.bar(mode_list_new, vals_mean_list, yerr=vals_std_list, color=color_list)
            ax.set_title(period)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([-0.5, 0, 0.5])
            ax.set_xticklabels(mode_list_new, rotation=45)
            flag += 1
        plt.tight_layout()
        plt.show()


class Monte_carlo_CV_moderate:
    def __init__(self):
        pass
    def run(self):
        # self.cal_monte_carlo_wet()
        # self.cal_monte_carlo_dry()
        # self.cal_monte_carlo_heat()
        # self.cal_monte_carlo_cold()
        # self.cal_monte_carlo_heat_compound()
        # self.cal_monte_carlo_cold_compound()
        self.check_result()
        self.difference()
        # self.contribution()
        pass
    pass

    def cal_monte_carlo_wet(self):
        outdir = join(result_root, 'monte_carlo_CV_moderate')
        T.mk_dir(outdir)
        mode = 'wet'

        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]

            picked_wet_index = (precipitation >=0.5) & (precipitation < 1)

            selected_year = year_list[picked_wet_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_dry(self):
        mode = 'dry'

        outdir = join(result_root, 'monte_carlo_CV_moderate')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_preciptation = result_root + rf'zscore\\CRU.npy'
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            precipitation = dic_precip[pix]
            ##percentile<10 is the extreme dry year`

            picked_dry_index = (precipitation > -1) & (precipitation <= -0.5)

            selected_year = year_list[picked_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat(self):
        mode = 'heat'

        outdir = join(result_root, 'monte_carlo_CV_moderate')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmax = result_root + rf'zscore\\tmax.npy'

        dic_tmax = T.load_npy(f_tmax)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmax:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]

            picked_heat_index =(temp >= 0.5) & (temp < 1)

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold(self):
        mode = 'cold'

        outdir = join(result_root, 'monte_carlo_CV_moderate')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_tmin = result_root + rf'zscore\\tmin.npy'

        dic_tmin = T.load_npy(f_tmin)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            ##percentile<10 is the extreme dry year`

            picked_heat_index = (temp <= -0.5) & (temp > -1)

            selected_year = year_list[picked_heat_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_cold_compound(self):
        # mode = 'cold_wet'
        mode= 'cold_dry'

        outdir = join(result_root, 'monte_carlo_CV_moderate')
        T.mk_dir(outdir)

        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmin = T.load_npy(f_tmin)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmin:
                continue
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            temp = dic_tmin[pix]
            precipitation = dic_precip[pix]
            ##selected cold and wet year/ cold and dry year

            # picked_cold_wet_index = (temp <= -0.5) & (temp > -1) & (precipitation >= 0.5) & (precipitation < 1)
            picked_cold_dry_index = (temp <= -0.5) & (temp > -1) & (precipitation <= -0.5) & (precipitation > -1)

            selected_year = year_list[picked_cold_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std

    def cal_monte_carlo_heat_compound(self):
        # mode = 'heat_wet'
        mode='heat_dry'

        outdir = join(result_root, 'monte_carlo_CV_moderate')
        T.mk_dir(outdir)

        f_tmax = result_root + rf'zscore\\tmax.npy'
        f_preciptation = result_root + rf'zscore\\CRU.npy'

        dic_tmax = T.load_npy(f_tmax)
        dic_precip = T.load_npy(f_preciptation)

        f_lai = rf'D:\Project4\Result\Detrend\detrend_raw\\LAI4g.npy'

        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []

        outf = join(outdir, f'monte_carlo_{mode}_year_LAI.npy')

        for pix in tqdm(dic_lai):
            if not pix in dic_tmax:
                continue
            if not pix in dic_precip:
                continue

            lai = dic_lai[pix]

            temp = dic_tmax[pix]

            precipitation = dic_precip[pix]

            ##selected heat wet and heat dry year

            # picked_heat_wet_index = (temp >= 0.5) & (temp < 1) & (precipitation >= 0.5) & (precipitation < 1)
            picked_heat_dry_index = (temp >= 0.5) & (temp < 1) & (precipitation <= -0.5) & (precipitation > -1)

            selected_year = year_list[picked_heat_dry_index]
            params = (pix, lai, year_list, selected_year)
            params_list.append(params)
        result = MULTIPROCESS(self.kernel_cal_monte_carlo_CV, params_list).run(process=14)
        # print(result)
        pickle.dump(result, open(outf, 'wb'))
        # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
        # plt.plot(range(len(lai_copy)), lai_copy,label='random')
        # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
        # plt.plot(range(len(lai)), lai, label='origin')
        # plt.scatter(range(len(lai)), lai, label='origin')
        # plt.legend()
        # plt.show()
        # plt.hist(slope_list,bins=30)
        # plt.show()
        ## calculate the mean and std of the slope

        # result_dict_slope[pix] = mean
        # result_dict_slope_std[pix] = std



    def kernel_cal_monte_carlo_CV(self, params):
        n = 100
        pix, lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        CV_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)

            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            CV = np.nanstd(lai_copy) / np.nanmean(lai_copy) * 100
            CV_list.append(CV)

        mean = np.nanmean(CV_list)
        std = np.nanstd(CV_list)
        return (pix, mean, std)

        pass

    def check_result(self):

        fdir = rf'D:\Project4\Result\monte_carlo_CV_moderate\\'
        for f in os.listdir(fdir):
            if not 'npy' in f:
                continue
            if 'slope' in f:
                continue
            if 'std' in f:
                continue

            fpath = join(fdir, f)
            result_dict = np.load(fpath, allow_pickle=True)
            spatial_dict1 = {}
            spatial_dict2 = {}
            # for i in result_dict:
            #     print(i)
            #     exit()
            for pix, slope, std in result_dict:
                # if np.isnan(slope):
                #     continue
                # print(pix,slope,std)

                spatial_dict1[pix] = slope
                spatial_dict2[pix] = std
            arr1 = D.pix_dic_to_spatial_arr(spatial_dict1)
            arr2 = D.pix_dic_to_spatial_arr(spatial_dict2)
            # plt.imshow(arr1)
            # plt.colorbar()
            # plt.show()
            ##save
            D.arr_to_tif(arr1, fpath.replace('.npy', '_slope.tif'))
            D.arr_to_tif(arr2, fpath.replace('.npy', '_std.tif'))
            np.save(fpath.replace('.npy', '_slope.npy'), spatial_dict1)
            np.save(fpath.replace('.npy', '_std.npy'), spatial_dict2)


    def difference(self): ### calculate the difference of the slope the real and scenario
        # NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        # array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        # landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        # crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        # MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        # dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        f_real_trend = result_root+rf'coefficient_of_variation\\LAI4g.npy.tif'
        array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real_trend)
        fdir=rf'D:\Project4\Result\monte_carlo_CV_moderate\\'
        for f in os.listdir(fdir):
            if not 'slope' in f:
                continue
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            # print(fpath)
            array_scenario, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real[array_real ==0] = np.nan
            array_scenario[array_scenario == 0] = np.nan
            array_diff = array_real - array_scenario
            array_diff[array_diff ==0] = np.nan

            fname_1=f.replace('slope','CV')
            fname=fname_1.replace('monte_carlo_','')

            outf=join(fdir,'difference', fname)
            T.mk_dir(join(fdir,'difference'))
            dic_arr_difference = D.spatial_arr_to_dic(array_diff)
            spatial_dict = {}
            for pix in tqdm(dic_arr_difference):
                r,c = pix


                # if not pix in dic_modis_mask:
                #     continue
                # if dic_modis_mask[pix] == 12:
                #     continue
                # landcover_value = crop_mask[pix]
                # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                #     continue
                spatial_dict[pix] = dic_arr_difference[pix]
            array_diff_new = D.pix_dic_to_spatial_arr(spatial_dict)
            # array_diff_new = array_diff_new * array_mask
            array_diff_new[array_diff_new > 99] = np.nan
            array_diff_new[array_diff_new < -99] = np.nan


            D.arr_to_tif(array_diff_new, outf)

    def contribution(self):
        fdir=rf'D:\Project4\Result\monte_carlo_CV_extreme\\difference\\'
        f_real=rf'D:\Project4\Result\coefficient_of_variation\\LAI4g.npy.tif'
        outdir=result_root+rf'monte_carlo_CV_extreme\\contribution\\'
        T.mk_dir(outdir,force=1)
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            array_diff, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real)
            # array_diff[array_diff == 0] = np.nan
            # array_real[array_real == 0] = np.nan
            array_diff = np.abs(array_diff)
            array_real = np.abs(array_real)
            array_contribution = array_diff/array_real*100
            array_contribution[array_contribution==100]=np.nan

            outf=join(outdir,f.replace('diff','contribution'))
            D.arr_to_tif(array_contribution, outf)


        pass

    def barplot(self):

        dff = rf'D:\Project3\Result\monte_carlo\Dataframe\monte_carlo.df'
        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # df=df[df['continent']=='Australia']
        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']>0]
        df = df.dropna()
        dry_color_list = ['peachpuff', 'orange', 'darkorange', 'chocolate', 'saddlebrown']

        wet_color_list = ['lightblue', 'cyan', 'deepskyblue', 'dodgerblue', 'navy']
        color_list = wet_color_list + dry_color_list[::-1]
        # df=df[df['partial_corr_GPCC']<0.4]
        df = df.dropna()
        # print(df.columns)
        # exit()
        #### plt. bar plot for the wet and dry year all the period

        result_period_dic = {}
        mode_list = ['extreme', 'moderate', 'mild', ]
        patterns = ['wet', 'dry']

        period_list = ['1982_2000', '2001_2020', '1982_2020']

        for period in period_list:

            lai_raw = df[f'LAI4g_{period}_trend'].to_list()
            lai_raw = np.array(lai_raw)
            lai_raw_mean = np.nanmean(lai_raw)
            lai_raw_std = np.nanstd(lai_raw)

            result_dic_pattern = {}
            for pattern in patterns:
                result_dic_mode = {}
                for mode in mode_list:
                    vals = df[f'{pattern}_year_trend_{mode}_{period}'].to_list()
                    vals_array = np.array(vals)
                    vals_mean = np.nanmean(vals_array)
                    vals_std = np.nanstd(vals_array)
                    result_dic_mode[mode] = (vals_mean, vals_std)
                ##add the mean of the raw data
                vals_mean = np.nanmean(lai_raw)
                vals_std = np.nanstd(lai_raw)
                result_dic_mode['raw'] = (vals_mean, vals_std)

                result_dic_pattern[pattern] = result_dic_mode
            result_period_dic[period] = result_dic_pattern

        ##plot
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        flag = 0
        for period in period_list:
            ax = axs.ravel()[flag]
            result_dic_pattern = result_period_dic[period]
            vals_mean_list = []
            vals_std_list = []
            for pattern in patterns:
                result_dic_mode = result_dic_pattern[pattern]

                for mode in mode_list:
                    vals_mean, vals_std = result_dic_mode[mode]
                    vals_mean_list.append(vals_mean)
                    vals_std_list.append(vals_std)
            mode_list_new = [f'{mode}_{pattern}' for pattern in patterns for mode in mode_list]

            ax.bar(mode_list_new, vals_mean_list, yerr=vals_std_list, color=color_list)
            ax.set_title(period)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([-0.5, 0, 0.5])
            ax.set_xticklabels(mode_list_new, rotation=45)
            flag += 1
        plt.tight_layout()
        plt.show()




class Calculate_extreme_event:
    def __init__(self):
        pass
    def run(self):
        self.check_histogram()
        self.cal_extreme_events_frequency()

        pass

    def check_histogram(self):
        f_preciptation = result_root + rf'zscore\\CRU.npy'
        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_tmax = result_root + rf'zscore\\tmax.npy'
        dic_precip = T.load_npy(f_preciptation)

        precipitation_list = []
        tmin_list = []
        tmax_list = []
        for pix in dic_precip:
            precipitation = dic_precip[pix]
            precipitation_list.append(precipitation)

        precipitation_arr = np.array(precipitation_list)
        precipitation_arr=precipitation_arr.flatten()
        ##remove nan
        precipitation_arr=precipitation_arr[~np.isnan(precipitation_arr)]
        ## calculate number of precitation zscore >1.5
        index= precipitation_arr >= 1
        index=precipitation_arr<= -1
        print(np.sum(index)/len(precipitation_arr))
        # exit()

        plt.hist(precipitation_arr, bins=100)
        plt.show()
        # exit()


        precip_90 = np.percentile(precipitation_arr, 90)
        precip_10 = np.percentile(precipitation_arr, 10)

        dic_tmin = T.load_npy(f_tmin)

        for pix in dic_tmin:
            tmin = dic_tmin[pix]
            tmin_list.append(tmin)
        tmin_arr = np.array(tmin_list)

        tmin_arr=tmin_arr.flatten()
        tmin_arr=tmin_arr[~np.isnan(tmin_arr)]
        index=tmin_arr>=1
        index=tmin_arr<=-1
        print(np.sum(index)/len(tmin_arr))


        plt.hist(tmin_arr, bins=100)
        plt.show()


        dic_tmax = T.load_npy(f_tmax)

        for pix in dic_tmin:
            tmax = dic_tmax[pix]
            tmax_list.append(tmax)
        tmax_arr = np.array(tmax_list)
        tmax_arr=tmax_arr.flatten()
        tmax_arr=tmax_arr[~np.isnan(tmax_arr)]
        print(np.sum(tmax_arr>=1)/len(tmax_arr))

        plt.hist(tmax_arr, bins=100)
        plt.show()
        # exit()



        pass

    def cal_extreme_events_frequency(self):

        # landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        # crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        # MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        # dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f_preciptation = result_root + rf'zscore\\CRU.npy'
        f_tmin = result_root + rf'zscore\\tmin.npy'
        f_tmax = result_root + rf'zscore\\tmax.npy'
        dic_precip = T.load_npy(f_preciptation)
        dic_tmin = T.load_npy(f_tmin)
        dic_tmax = T.load_npy(f_tmax)



        spatial_wet_dic={}
        spatial_dry_dic={}
        spatial_heat_dic={}
        spatial_cold_dic={}
        spatial_wet_cold_dic={}
        spatial_wet_heat_dic={}
        spatial_dry_cold_dic={}
        spatial_dry_heat_dic={}




        for pix in dic_precip:

            # landcover_value = crop_mask[pix]
            # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
            #     continue
            # if dic_modis_mask[pix] == 12:
            #     continue

            precipitation=dic_precip[pix]
            if not pix in dic_tmin:
                continue
            if not pix in dic_tmax:
                continue
            tmin=dic_tmin[pix]
            tmax=dic_tmax[pix]


            wet_index=precipitation>=1
            dry_index=precipitation<=-1
            heat_index=tmax>=1
            cold_index=tmin<=-1
            wet_cold_index=wet_index & cold_index
            wet_heat_index=wet_index & heat_index
            dry_cold_index=dry_index & cold_index
            dry_heat_index=dry_index & heat_index

            wet_year=np.sum(wet_index)
            dry_year=np.sum(dry_index)
            heat_year=np.sum(heat_index)
            cold_year=np.sum(cold_index)
            wet_cold_year=np.sum(wet_cold_index)
            wet_heat_year=np.sum(wet_heat_index)
            dry_cold_year=np.sum(dry_cold_index)
            dry_heat_year=np.sum(dry_heat_index)
            spatial_dry_dic[pix]=dry_year
            spatial_wet_dic[pix]=wet_year
            spatial_heat_dic[pix]=heat_year
            spatial_cold_dic[pix]=cold_year
            spatial_wet_cold_dic[pix]=wet_cold_year
            spatial_wet_heat_dic[pix]=wet_heat_year
            spatial_dry_cold_dic[pix]=dry_cold_year
            spatial_dry_heat_dic[pix]=dry_heat_year
        outdir=result_root + rf'extreme_event\\'
        Tools().mk_dir(outdir, force=True)
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_wet_dic),result_root + rf'extreme_event\wet_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dry_dic),result_root + rf'extreme_event\dry_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_heat_dic),result_root + rf'extreme_event\heat_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_cold_dic),result_root + rf'extreme_event\cold_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_wet_cold_dic),result_root + rf'extreme_event\wet_cold_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_wet_heat_dic),result_root + rf'extreme_event\wet_heat_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dry_cold_dic),result_root + rf'extreme_event\dry_cold_year.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dry_heat_dic),result_root + rf'extreme_event\dry_heat_year.tif')


        pass

class Machine_learning_cluster:
    def __init__(self):

        pass
    def run(self):
        self.cluster()

    def cluster(self):
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import SpectralClustering
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.preprocessing import StandardScaler

        fdir= rf'D:\Project4\Result\monte_carlo_CV_extreme\\difference\\'
        X_list=[]
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'xml' in f:
                continue
            fpath=join(fdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)


            X_list.append(array)
        flattened_map_list = [x.flatten() for x in X_list]

        X = np.array(flattened_map_list).T


        # input data with 8 features

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(X)



        # KMeans Clustering
        n_components = 4  # Number of clusters
        spectral  = SpectralClustering(n_clusters=n_components, affinity='nearest_neighbors', n_neighbors=10, random_state=0)

        labels = spectral.fit_predict(normalized_data)

        # Plotting
        clustered_map = labels.reshape(array.shape)

        plt.figure(figsize=(10, 6))
        plt.imshow(clustered_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Cluster Label')
        plt.title('Spatial Clustering using GMM')
        plt.show()

    pass

class Probability_density:
    def __init__(self):
        pass
    def run(self):
        # self.rename_Global_Ecological_Zones()
        # self.rename_MODIC_LUCC()
        # self.rename_MODIS_LUCC_sixteen()
        # self.rename_IPCC_climate_zone_v4()
        # self.rename_IPCC_climate_zone_v3()
        # self.rename_MODIC_LUCC()
        # self.aggragated_IPCC_climate_zone()
        # self.plot_boxplot_climate_LUCC()   ## all events plot together for each climate zone and LUCC
        self.plot_boxplot_LUCC()  #no climate regions but all events together

        # self.plot_boxplot_LUCC_climate() ### each pft and  event across all the climate zone
        # self.plot_boxplot_climate()
        # self.plot_boxplot_LUCC_seperately() # all climate regions together



        # self.plot_probability_density()
        pass
    def rename_Global_Ecological_Zones(self):
        dic_new = {41:'Boreal coniferous forest',
                   43:'Boreal mountain system',
                   42:'Boreal tundra woodland',
                    50:'Polar',
                    24:'Subtropical desert',
                   22:'Subtropical dry forest',
                   21:'Subtropical humid forest',
                     25:'Subtropical mountain system',
                     23:'Subtropical steppe',
                        32:'Temperate continental forest',
                        34:'Temperate desert',
                        35:'Temperate mountain system',
                        31:'Temperate oceanic forest',
                        33:'Temperate steppe',
                        15:'Tropical desert',
                        13:'Tropical dry forest',
                        12:'Tropical moist forest',
                        16:'Tropical mountain system',
        11:'Tropical rainforest',
        14:'Tropical shrubland',
                   90:'water',

                   }

        f=rf'D:\Project4\Data\Base_data\Global_Ecological_Zone\tif\\gez_2010.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        dic=DIC_and_TIF().spatial_arr_to_dic(array)
        spatial_new={}

        for pix in dic:
            vals=dic[pix]
            print(vals)
            if vals<-99:
                spatial_new[pix]=np.nan
                continue


            spatial_new[pix]=dic_new[dic[pix]]

        outf=rf'D:\Project4\Data\Base_data\Global_Ecological_Zone\tif\\gez_2010_rename.npy'

        np.save(outf,spatial_new)

    def rename_MODIC_LUCC(self):
        dic_new = {1: 'Forests',
                     2: 'Shrubland',
                        3: 'Grassland',
                        4: 'Cropland',
                   }

        fdir = rf'D:\Project4\Data\Base_data\\MODIS_IGBP_WGS84\IGBP_reclass_resample\\'
        outdir=rf'D:\Project4\Data\Base_data\MODIS_IGBP_WGS84\\IGBP_reclass_resample_rename\\'
        Tools().mk_dir(outdir,force=True)
        for f in os.listdir(fdir):
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir,f))
            dic = DIC_and_TIF().spatial_arr_to_dic(array)
            spatial_new = {}

            for pix in dic:
                vals = dic[pix]
                # print(vals)
                if np.isnan(vals):
                    spatial_new[pix] = np.nan
                    continue
                if vals < -99:
                    spatial_new[pix] = np.nan
                    continue

                spatial_new[pix] = dic_new[dic[pix]]

            outf = join(outdir, f)

            np.save(outf, spatial_new)

            pass

    def rename_MODIS_LUCC_sixteen(self):
        fdir = rf'D:\Project4\Data\Base_data\MODIS_IGBP_WGS84\WGS84_resample\\'
        for f in os.listdir(fdir):

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic_new = {1: 'Evergreen Needleleaf Forests',
                         2: 'Evergreen Broadleaf Forests',
                            3: 'Deciduous Needleleaf Forests',
                            4: 'Deciduous Broadleaf Forests',
                            5: 'Mixed Forests',
                            6: 'Closed Shrublands',
                            7: 'Open Shrublands',
                            8: 'Woody Savannas',
                            9: 'Savannas',
                            10: 'Grasslands',
                            11: 'Permanent Wetlands',
                            12: 'Croplands',
                            13: 'Urban and Built-up',
                            14: 'Cropland/Natural Vegetation Mosaics',
                            15: 'Permanent Snow and Ice',
                            16: 'Barren',
                       17: 'Water Bodies',
                         255: 'Unclassified',
                       }
            spatial_new = {}

            for pix in dic:
                vals = dic[pix]
                if vals < -99:
                    dic[pix] = np.nan
                    continue
                if vals not in dic_new:
                    dic[pix] = np.nan
                    continue

                spatial_new[pix] = dic_new[dic[pix]]
            outdir= rf'D:\Project4\Data\Base_data\MODIS_IGBP_WGS84\WGS84_resample_rename\\'
            Tools().mk_dir(outdir,force=True)
            outf = join(outdir, f.replace('.tif', '.npy'))
            np.save(outf, spatial_new)

    def rename_IPCC_climate_zone_v4(self):
        f=rf'D:\Project4\Data\Base_data\IPCC_cliamte_zone\tif\\IPCC-WGI.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        dic=DIC_and_TIF().spatial_arr_to_dic(array)
        dic_new = {1: 'CIC', 2:'NWN',
        3:'NEN',
        4:'WNA',
        5:'CNA',
        6:'ENA',
        7:'NCA',
        8:'SCA',

        10:'NWS',
        11:'NSA',
        12:'NES',
        13:'SAM',
        14:'SWS',
        15:'SES',
        16:'SSA',
        17:'NEU',
        18:'WCE',
        19:'EEU',

        21:'SAH',
        22:'WAF',
        23:'CAF',
        24:'NEAF',
        25:'SEAF',
        26:'WSAF',
        27:'ESAF',
        28:'MDG',
        29:'RAR',
        30:'WSB',
        31:'ESB',
        32:'RFE',
        33:'WCA',
        34:'ECA',
        35:'TIB',
        36:'EAS',
        37:'ARP',
        38:'SAS',

        40:'NAU',
        41:'CAU',
        42:'EAU',
        43:'SAU',
        44:'NZ',
        45:'EAN',
        46:'WAN',
        }
        spatial_new = {}

        for pix in dic:
            vals=dic[pix]
            if vals<-99:
                dic[pix]=np.nan
                continue
            if vals not in dic_new:
                dic[pix]=np.nan
                continue

            spatial_new[pix]=dic_new[dic[pix]]
        outf=rf'D:\Project4\Data\Base_data\IPCC_cliamte_zone\tif\\IPCC-WGI_rename.npy'
        np.save(outf,spatial_new)

    def rename_IPCC_climate_zone_v3(self):
        f=rf'D:\Project4\Data\Base_data\IPCC_climate_zone_v3\\IPCC-WGI-v3.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        dic=DIC_and_TIF().spatial_arr_to_dic(array)
        dic_new = {1: 'ALA', 2:'AMZ',
        3:'CAM', 5:'CAS',
        6:'CEU', 7:'CGI',
        8:'CNA', 9:'EAF',
        10:'EAS', 11:'ENA',
        12:'MED', 13:'NAS',
        14:'NAU', 15:'NEB',
        16:'NEU', 17:'SAF',
        18:'SAH', 20:'SAU',
                   19:'SAS',
        21:'SEA',
         22:'SSA', 23:'TIB',
            24:'WAF', 25:'WAS',
            26:'WNA', 27:'WSA',

        }
        spatial_new = {}

        for pix in dic:
            vals=dic[pix]
            if vals<-99:
                dic[pix]=np.nan
                continue
            if vals not in dic_new:
                dic[pix]=np.nan
                continue

            spatial_new[pix]=dic_new[dic[pix]]
        outf=rf'D:\Project4\Data\Base_data\IPCC_climate_zone_v3\\IPCC-WGI-v3_rename.npy'
        np.save(outf,spatial_new)

    def aggragated_IPCC_climate_zone(self):
        f=rf'D:\Project4\Data\Base_data\IPCC_cliamte_zone\tif\\IPCC-WGI.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        dic=DIC_and_TIF().spatial_arr_to_dic(array)
        dic_new = {1: 'GIC', 2:'NWN',
        3:'NEN',
        4:'WNA',
        5:'CNA',
        6:'ENA',
        7:'NCA',
        8:'SCA',

        10:'NWS',
        11:'NSA',
        12:'NES',
        13:'SAM',
        14:'SWS',
        15:'SES',
        16:'SSA',
        17:'NEU',
        18:'WCE',
        19:'EEU',

        21:'SAH',
        22:'WAF',
        23:'CAF',
        24:'NEAF',
        25:'SEAF',
        26:'WSAF',
        27:'ESAF',
        28:'MDG',
        29:'RAR',
        30:'WSB',
        31:'ESB',
        32:'RFE',
        33:'WCA',
        34:'ECA',
        35:'TIB',
        36:'EAS',
        37:'ARP',
        38:'SAS',

        40:'NAU',
        41:'CAU',
        42:'EAU',
        43:'SAU',
        44:'NZ',
        45:'EAN',
        46:'WAN',
        }
        spatial_new = {}

        for pix in dic:
            vals=dic[pix]
            if vals<-99:
                dic[pix]=np.nan
                continue
            if vals not in dic_new:
                dic[pix]=np.nan
                continue

            spatial_new[pix]=dic_new[dic[pix]]
        outf=rf'D:\Project4\Data\Base_data\IPCC_cliamte_zone\tif\\IPCC-WGI_rename.npy'
        np.save(outf,spatial_new)













    def plot_boxplot_climate_LUCC(self): ### Fancy plot
        mode='CV'
        df=rf'E:\Project4\Result\monte_carlo_{mode}_extreme\Dataframe\monte_carlo_{mode}.df'
        df=T.load_df(df)
        ## seaborn plot

        IPCC_climate_zone_list = df['IPCC_classfication_v3'].unique()
        ##remove the nan from the list
        IPCC_climate_zone_list = IPCC_climate_zone_list[~pd.isnull(IPCC_climate_zone_list)]
        ## remove the water, barren and unclassified
        IPCC_climate_zone_list = ['ALA', 'WNA', 'CNA', 'ENA',
                                  'NEU', 'CEU', 'MED',
                                  'NAS',
                                  'WASt', 'CAS', 'EAS',
                                  'SAH', 'SAF',
                                  'AMZ', 'NEB', 'WSA', 'SSA',
                                  'WAF', 'EAF', 'SAF',
                                  'NAU', 'SAU',
                                  ]

        MODIS_LUCC_list=['Forests','Shrubland','Grassland','ALL'] ## remove the Cropland


        event_list=['cold_year_LAI','wet_year_LAI','cold_dry_year_LAI','cold_wet_year_LAI','dry_year_LAI','heat_year_LAI','heat_dry_year_LAI','heat_wet_year_LAI','all_extreme_year_LAI']


        IPCC_dic={}
        IPCC_list=[]

        for IPCC in IPCC_climate_zone_list:

            df_IPCC=df[df['IPCC_classfication_v3']==IPCC]

            df_IPCC = df_IPCC.dropna(subset=[f'LAI4g_{mode}'])
            if len(df_IPCC)==0:
                continue
            if len(df_IPCC)<500:
                continue
            print(IPCC,len(df_IPCC))
            IPCC_list.append(IPCC)

            LUCC_dic={}

            for LUCC in MODIS_LUCC_list:
                if LUCC=='ALL':
                    df_sub=df_IPCC
                else:

                    df_sub=df_IPCC[df_IPCC['MODIS_LUCC']==LUCC]
                event_dic={}
                for col in event_list:

                    col_name=col+f'_{mode}'

                    array=df_sub[col_name].to_numpy()
                    # if not 'LAI4g' in col:
                    array[array>5]=np.nan
                    array[array<-5]=np.nan
                    array=array[~np.isnan(array)]
                    if len(array) ==0:
                        continue
                    event_dic[col]=array
                LUCC_dic[LUCC]=event_dic
            IPCC_dic[IPCC]=LUCC_dic

        ##plot grouped by IPCC and LUCC
        fig, axs = plt.subplots(5, 4, figsize=(20,12))
        flag = 0

        ##plot stack bar as function of AI and period
        for IPCC in IPCC_list:

            ax = axs.ravel()[flag]
            result_dic_IPCC = IPCC_dic[IPCC]
            # pprint(result_dic_IPCC)
            x_list=[]
            y_list=[]
            hue_list=[]
            for LUCC in MODIS_LUCC_list:
                result_dic_LUCC = result_dic_IPCC[LUCC]
                for event in result_dic_LUCC:
                    array = result_dic_LUCC[event]
                    for val in array:
                        x_list.append(LUCC)
                        y_list.append(val)
                        hue_list.append(event)
            df_plot=pd.DataFrame({'x':x_list,'y':y_list,'hue':hue_list})
            # sns.violinplot(x='x',y='y',hue='hue',data=df_plot,ax=ax)
            # sns.violinplot(x='x', y='y', hue='hue', data=df_plot)
            sns.boxplot(x='x', y='y', hue='hue', data=df_plot,palette='Spectral_r',showfliers=False,ax=ax)

            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim(-3,3)
            ax.set_title(IPCC)
            ## set line y=0
            ax.axhline(0, color='grey', linewidth=1)

            ## remove legend
            ax.get_legend().remove()


            flag += 1
        plt.tight_layout()
        plt.show()


    def plot_boxplot_LUCC(self): ### no climate regions but all events together
        mode='CV'
        df=rf'E:\Project4\Result\monte_carlo_{mode}_extreme\Dataframe\monte_carlo_{mode}.df'
        df=T.load_df(df)
        ## seaborn plot

        MODIS_LUCC_list=['Forests','Shrubland','Grassland','ALL']


        event_list=['cold_year_LAI','wet_year_LAI','cold_dry_year_LAI','cold_wet_year_LAI','dry_year_LAI','heat_year_LAI','heat_dry_year_LAI','heat_wet_year_LAI','all_extreme_year_LAI']



        LUCC_dic={}

        for LUCC in MODIS_LUCC_list:
            if LUCC=='ALL':
                df_sub= df
            else:

                df_sub=df[df['MODIS_LUCC']==LUCC]
            event_dic={}
            for col in event_list:

                col_name=col+f'_{mode}'

                array=df_sub[col_name].to_numpy()
                # if not 'LAI4g' in col:
                array[array>5]=np.nan
                array[array<-5]=np.nan
                array=array[~np.isnan(array)]
                if len(array) ==0:
                    continue
                event_dic[col]=array
            LUCC_dic[LUCC]=event_dic

        x_list = []
        y_list = []
        hue_list = []

        for LUCC in MODIS_LUCC_list:

            result_dic_LUCC = LUCC_dic[LUCC]

            for event in result_dic_LUCC:
                array = result_dic_LUCC[event]
                for val in array:
                    x_list.append(LUCC)
                    y_list.append(val)
                    hue_list.append(event)
            df_plot=pd.DataFrame({'x':x_list,'y':y_list,'hue':hue_list})

        sns.boxplot(x='x', y='y', hue='hue', data=df_plot,palette='Spectral_r',showfliers=False)

        plt.xlabel('')
        plt.ylabel('CV %')
        plt.ylim(-3, 3)

        ## set line y=0
        plt.axhline(0, color='grey', linewidth=1)

        plt.show()





    def plot_boxplot_LUCC_climate(self):  ## each event has a subplot
        df=rf'E:\Project4\Result\monte_carlo_trend_extreme\Dataframe\monte_carlo_trend.df'
        df=T.load_df(df)
        ## seaborn plot

        IPCC_climate_zone_list = df['IPCC_classfication_v3'].unique()
        ##remove the nan from the list
        IPCC_climate_zone_list = IPCC_climate_zone_list[~pd.isnull(IPCC_climate_zone_list)]
        ## remove the water, barren and unclassified
        IPCC_climate_zone_list = IPCC_climate_zone_list[
            ~np.isin(IPCC_climate_zone_list, ['ALA', 'CGI', 'TIB', 'SEA', 'CAM', ])]
        IPCC_climate_zone_list=['ALA', 'WNA', 'CNA','ENA',
                               'NEU','CEU','MED',
                                'NAS',
                                'WASt','CAS','EAS',
                                'SAH','SAF',
                                'AMZ','NEB','WSA','SSA',
                                'WAF','EAF','SAF',
                                'NAU','SAU',
                                'ALL'
        ]

        MODIS_LUCC_list=['Forests','Shrubland','Grassland','Cropland',]


        event_list=['cold_year_LAI','wet_year_LAI','cold_dry_year_LAI','cold_wet_year_LAI','dry_year_LAI','heat_year_LAI','heat_dry_year_LAI','heat_wet_year_LAI']


        for event in event_list:
            LUCC_dic={}
            for LUCC in MODIS_LUCC_list:

                df_sub=df[df['MODIS_LUCC']==LUCC]
                df_sub = df_sub.dropna(subset=['LAI4g_trend'])
                if len(df_sub)==0:
                    continue
                # if len(df_sub)<500:
                #     continue
                print(LUCC,len(df_sub))
                IPCC_dic={}


                for IPCC in IPCC_climate_zone_list:
                    if IPCC=='ALL':
                        df_IPCC=df_sub
                    else:


                        df_IPCC=df_sub[df_sub['IPCC_classfication_v3']==IPCC]
                    # if len(df_IPCC)==0:
                    #     continue
                    # if len(df_IPCC)<500:
                    #     continue
                    # IPCC_list.append(IPCC)

                    col_name=event+'_trend'

                    array=df_IPCC[col_name].to_numpy()
                    # if not 'LAI4g' in col:
                    array[array>5]=np.nan
                    array[array<-5]=np.nan
                    array=array[~np.isnan(array)]
                    if len(array) ==0:
                        continue
                    IPCC_dic[IPCC]=array

                LUCC_dic[LUCC]=IPCC_dic

            ##plot grouped by IPCC and LUCC
            fig, axs = plt.subplots(4, 1, figsize=(12,12))
            flag = 0
            for LUCC in MODIS_LUCC_list:

                ax = axs.ravel()[flag]
                result_dic_LUCC = LUCC_dic[LUCC]
                # pprint(result_dic_IPCC)
                x_list=[]
                y_list=[]

                for IPCC in result_dic_LUCC:
                    array = result_dic_LUCC[IPCC]
                    x_list.append(array)
                    y_list.append(IPCC)
                df_new=pd.DataFrame(x_list).T
                ##  median
                sns.boxplot(data=df_new,palette='Spectral_r',showfliers=False,ax=ax)
                ax.set_xticklabels(y_list, rotation=0)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_ylim(-0.5,0.5)
                ax.set_title(LUCC)
                ## set line y=0
                ax.axhline(0, color='grey', linewidth=1)
                flag=flag+1

            plt.suptitle(event)
            plt.show()






    def plot_boxplot_climate(self):
        df=rf'D:\Project4\Result\monte_carlo_trend_extreme\Dataframe\monte_carlo_trend.df'
        df=T.load_df(df)
        ## seaborn plot


        IPCC_climate_zone_list=df['IPCC_classfication_v3'].unique()
        ##remove the nan from the list
        IPCC_climate_zone_list=IPCC_climate_zone_list[~pd.isnull(IPCC_climate_zone_list)]
        ## remove the water, barren and unclassified
        IPCC_climate_zone_list=IPCC_climate_zone_list[~np.isin(IPCC_climate_zone_list,['ALA','CGI','TIB','SEA','CAM',])]


        event_list=['cold_year_LAI','wet_year_LAI','cold_dry_year_LAI','cold_wet_year_LAI','dry_year_LAI','heat_year_LAI','heat_dry_year_LAI','heat_wet_year_LAI']


        IPCC_dic={}
        IPCC_list=[]

        for IPCC in IPCC_climate_zone_list:
            df_IPCC=df[df['IPCC_classfication_v3']==IPCC]

            df_IPCC = df_IPCC.dropna(subset=['LAI4g_trend'])
            if len(df_IPCC)==0:
                continue
            if len(df_IPCC)<500:
                continue
            print(IPCC,len(df_IPCC))
            IPCC_list.append(IPCC)

            event_dic={}


            for col in event_list:

                col_name=col+'_trend'

                array=df_IPCC[col_name].to_numpy()
                # if not 'LAI4g' in col:
                array[array>5]=np.nan
                array[array<-5]=np.nan
                array=array[~np.isnan(array)]
                if len(array) ==0:
                    continue
                event_dic[col]=array
            IPCC_dic[IPCC]=event_dic



        ##plot grouped by IPCC and LUCC
        fig, axs = plt.subplots(5, 4, figsize=(12,12))
        flag = 0

        ##plot stack bar as function of AI and period
        for IPCC in IPCC_list:

            ax = axs.ravel()[flag]
            result_dic_IPCC = IPCC_dic[IPCC]
            # pprint(result_dic_IPCC)
            x_list=[]
            y_list=[]
            hue_list=[]
            for event in result_dic_IPCC:
                array = result_dic_IPCC[event]
                for val in array:
                    x_list.append(IPCC)
                    y_list.append(val)
                    hue_list.append(event)
            df_plot=pd.DataFrame({'x':x_list,'y':y_list,'hue':hue_list})
            # sns.violinplot(x='x',y='y',hue='hue',data=df_plot,ax=ax)
            # sns.violinplot(x='x', y='y', hue='hue', data=df_plot)
            sns.boxplot(x='x', y='y', hue='hue', data=df_plot,palette='Spectral_r',showfliers=False,ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ## set line y=0
            ax.axhline(0, color='grey', linewidth=1)

            ## remove legend
            ax.get_legend().remove()


            # plt.title(IPCC)
            flag += 1
        plt.show()

    def plot_boxplot_LUCC_seperately(self): ## all climate regions together
        df=rf'D:\Project4\Result\monte_carlo_trend_extreme\Dataframe\monte_carlo_trend.df'
        df=T.load_df(df)
        ## seaborn plot



        ##remove the nan from the list
        MODIS_LUCC_list = df['MODIS_LUCC_sixteen'].unique()
        ##remove the nan from the list
        MODIS_LUCC_list = MODIS_LUCC_list[~pd.isnull(MODIS_LUCC_list)]
        ##remove the water, barren and unclassified
        MODIS_LUCC_list = MODIS_LUCC_list[~np.isin(MODIS_LUCC_list,
                                                   ['Water Bodies', 'Barren', 'Unclassified', 'Permanent Snow and Ice',
                                                    'Urban and Built-up',
                                                    'Cropland/Natural Vegetation Mosaics', 'Permanent Wetlands'])]

        event_list=['cold_year_LAI','wet_year_LAI','cold_dry_year_LAI','cold_wet_year_LAI','dry_year_LAI','heat_year_LAI','heat_dry_year_LAI','heat_wet_year_LAI']


        IPCC_dic={}
        IPCC_list=[]

        for LUCC in MODIS_LUCC_list:
            df_LUCC=df[df['MODIS_LUCC_sixteen']==LUCC]

            df_LUCC = df_LUCC.dropna(subset=['LAI4g_trend'])
            if len(df_LUCC)==0:
                continue
            if len(df_LUCC)<500:
                continue
            print(LUCC,len(df_LUCC))
            IPCC_list.append(LUCC)

            event_dic={}


            for col in event_list:

                col_name=col+'_trend'

                array=df_LUCC[col_name].to_numpy()
                array[array>5]=np.nan
                array[array<-5]=np.nan
                array=array[~np.isnan(array)]
                if len(array) ==0:
                    continue
                event_dic[col]=array
            IPCC_dic[LUCC]=event_dic



        ##plot grouped by IPCC and LUCC
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        flag = 0

        ##plot stack bar as function of AI and period
        for IPCC in IPCC_list:

            ax = axs.ravel()[flag]
            result_dic_IPCC = IPCC_dic[IPCC]
            # pprint(result_dic_IPCC)
            x_list=[]
            y_list=[]
            hue_list=[]
            for event in result_dic_IPCC:
                array = result_dic_IPCC[event]
                for val in array:
                    x_list.append(IPCC)
                    y_list.append(val)
                    hue_list.append(event)
            df_plot=pd.DataFrame({'x':x_list,'y':y_list,'hue':hue_list})
            # sns.violinplot(x='x',y='y',hue='hue',data=df_plot,ax=ax)
            # sns.violinplot(x='x', y='y', hue='hue', data=df_plot)
            sns.boxplot(x='x', y='y', hue='hue', data=df_plot,palette='Spectral_r',showfliers=False,ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ## set line y=0
            ax.axhline(0, color='grey', linewidth=1)

            ## remove legend
            ax.get_legend().remove()


            # plt.title(IPCC)
            flag += 1
        plt.show()






    def plot_probability_density(self):
        df=result_root+rf'monte_carlo_trend_extreme\Dataframe\monte_carlo_trend.df'
        df=T.load_df(df)
        ## seaborn plot


        MODIS_LUCC_list=df['MODIS_LUCC'].unique()
        ##remove the nan from the list
        MODIS_LUCC_list=MODIS_LUCC_list[~pd.isnull(MODIS_LUCC_list)]
        ##remove the water, barren and unclassified
        MODIS_LUCC_list=MODIS_LUCC_list[~np.isin(MODIS_LUCC_list,['Water Bodies','Barren','Unclassified','Permanent Snow and Ice','Urban and Built-up',
                                                                  'Cropland/Natural Vegetation Mosaics','Permanent Wetlands'])]


        event_list=['cold_year_LAI','wet_year_LAI','cold_dry_year_LAI','cold_wet_year_LAI','dry_year_LAI','heat_year_LAI','heat_dry_year_LAI','heat_wet_year_LAI']



        for event in event_list:

            df_event=df.dropna(subset=['LAI4g_trend'])
            if len(df_event)==0:
                continue

            for LUCC in MODIS_LUCC_list:

                df_sub=df_event[df_event['MODIS_LUCC_sixteen']==LUCC]
                values=df_sub[event+'_trend'].to_numpy()
                values[values>1]=np.nan
                values[values<-1]=np.nan
                values=values[~np.isnan(values)]
                if len(values) ==0:
                    continue
                sns.kdeplot(values,label=LUCC)
            plt.legend()
            plt.title(event)
            plt.show()










class build_dataframe():


    def __init__(self):


        self.this_class_arr = result_root + rf'\monte_carlo_trend_extreme\\Dataframe\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'monte_carlo_trend.df'


        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)
        # df=self.build_df(df)
        # df=self.build_df_monthly(df)
        # df=self.append_attributes(df)  ## 加属性
        # df=self.append_cluster(df)  ## 加属性
        # df=self.append_value(df)


        df=self.add_trend_to_df(df)
        # df=self.add_IPCC_classfication(df)
        # df=self.add_MODIS_LUCC(df)
        # df=self.add_MODIS_LUCC_sixteen(df)
        # df=self.add_IPCC_climate_zone(df)


        # df=self.add_AI_classfication(df)




        T.save_df(df, self.dff)

        self.__df_to_excel(df, self.dff)

    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass
    def build_df(self, df):

        fdir =result_root+ rf'\relative_change\OBS_LAI_extend\\'
        all_dic= {}
        for f in os.listdir(fdir):


            fname= f.split('.')[0]
            if fname not in ['LAI4g','GPCC', 'tmax','VPD','GLEAM_SMroot','CRU','leaf_area']:
                continue

            fpath=fdir+f

            dic = T.load_npy(fpath)
            key_name=fname
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df

    def build_df_monthly(self, df):


        fdir = result_root+rf'extract_GS_return_monthly_data\individual_month_relative_change\X\\'
        all_dic= {}

        for fdir_ii in os.listdir(fdir):

            dic=T.load_npy(fdir+fdir_ii)

            key_name=fdir_ii.split('.')[0]
            all_dic[key_name] = dic
                # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df



    def append_attributes(self, df):  ## add attributes
        fdir =  result_root + rf'\relative_change\OBS_LAI_extend\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            if not 'leaf_area' in f:
                continue

            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic=T.load_npy(fdir+f)
            key_name = f.split('.')[0]
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df=T.add_spatial_dic_to_df(df,dic,key_name)
        return df


    def append_cluster(self, df):  ## add attributes
        dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,

                     'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,

                     'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}

        #### reverse
        dic_label = {v: k for k, v in dic_label.items()}


        fdir = result_root+rf'Dataframe\anomaly_trends\\'
        for f in os.listdir(fdir):
            if not f.endswith('tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)

            # array=np.load(fdir+f)
            dic = DIC_and_TIF().spatial_arr_to_dic(array)

            key_name='label'
            for k in dic:
                if dic[k] <-99:
                    continue
                dic[k]=dic_label[dic[k]]

            df=T.add_spatial_dic_to_df(df,dic,key_name)

        return df






    def append_value(self, df):  ##补齐
        fdir = result_root + rf'\\anomaly\OBS_extend\\'
        col_list=[]
        for f in os.listdir(fdir):

            if not f.endswith('.npy'):
                continue


            col_name=f.split('.')[0]
            if col_name not in ['LAI4g','GPCC', 'tmax','VPD','GLEAM_SMroot',]:
                continue
            col_list.append(col_name)

        for col in col_list:
            vals_new=[]

            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
                pix = row['pix']
                r, c = pix
                vals=row[col]
                if type(vals)==float:
                    vals_new.append(np.nan)
                    continue
                vals=np.array(vals)
                # if len(vals)==23:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     # print(len(vals))
                # elif len(vals)==38:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     print(len(vals))
                if len(vals)==38:
                    vals=np.append(vals,np.nan)
                    vals_new.append(vals)

                vals_new.append(vals)

                # exit()
            df[col]=vals_new

        return df

        pass


    def foo1(self, df):

        f = result_root + rf'\moving_window_extraction\dry_year_moving_window_extraction\\dry_year_moving_window_extraction_lai.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)


        dic = T.load_npy(f)


        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1982
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y)
                y += 1


        df['pix'] = pix_list

        df['year'] = year
        # df['window'] = 'VPD_LAI4g_00'
        df['LAI4g'] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'\monte_carlo_trend_extreme\difference\\cold_dry_year_LAI_trend.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(f)
        # val_array[val_array<-99]=np.nan
        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        # plt.imshow(val_array)
        # plt.colorbar()
        # plt.show()

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            val = val_dic[pix]
            if np.isnan(val):
                continue
            pix_list.append(pix)
        df['pix'] = pix_list
        T.print_head_n(df)


        return df

    def add_trend_to_df(self, df):
        fdir=result_root+rf'\monte_carlo_trend_extreme\difference\\'
        for f in os.listdir(fdir):
            # print(f)
            # exit()
            if not 'tif' in f:
                continue
            if not 'all_extreme' in f:
                continue



            if not f.endswith('.tif'):
                continue

            variable = (f.split('.')[0])

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)
            # val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                if val > 99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f'{f_name}'] = val_list


        return df

    def add_IPCC_classfication(self, df):

        f = data_root + rf'Base_data\IPCC_climate_zone_v3\\IPCC-WGI-v3_rename.npy'

        val_dic=T.load_npy(f)


        f_name = 'IPCC_classfication_v3'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # if val < -99:
            #     val_list.append(np.nan)
            #     continue
            val_list.append(val)
        df[f'{f_name}'] = val_list

        return df

    def add_MODIS_LUCC(self, df):

        f = data_root + rf'\Base_data\\MODIS_IGBP_WGS84\\IGBP_reclass_resample_rename\\2010_01_01_resample.tif.npy'

        val_dic=T.load_npy(f)


        f_name = 'MODIS_LUCC'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # if val < -99:
            #     val_list.append(np.nan)
            #     continue
            val_list.append(val)
        df[f'{f_name}'] = val_list

        return df

    def add_MODIS_LUCC_sixteen(self, df):
        f = data_root + rf'Base_data\MODIS_IGBP_WGS84\WGS84\WGS84_resample_rename\\2010_01_01_resample.npy'

        val_dic = T.load_npy(f)

        f_name = 'MODIS_LUCC_sixteen'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # if val < -99:
            #     val_list.append(np.nan)
            #     continue
            val_list.append(val)
        df[f'{f_name}'] = val_list

        return df


    def add_IPCC_climate_zone(self, df):

        f = data_root + rf'\Base_data\\IPCC_cliamte_zone\tif\\IPCC-WGI_rename.npy'

        val_dic=T.load_npy(f)


        f_name = 'IPCC_climate_zone'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # if val < -99:
            #     val_list.append(np.nan)
            #     continue
            val_list.append(val)
        df[f'{f_name}'] = val_list

        return df


    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_max_trend_to_df(self, df):

        fdir = data_root + rf'/Base_data/lc_trend/'
        for f in (os.listdir(fdir)):
            # print()
            if not 'max_trend' in f:
                continue
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue

            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val * 20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list

        return df

    pass
    def add_lat_lon_to_df(self, df):
        D=DIC_and_TIF(pixelsize=0.25)
        df=T.add_lon_lat_to_df(df,D)
        return df













    def rename_columns(self, df):
        df = df.rename(columns={'Inversion_relative_change': 'Inversion',


                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[rf'monte_carlo_relative_change_normal_slope',

                              ])
        return df



    def add_NDVI_mask(self, df):
        f = data_root + rf'/Base_data/NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df




    def add_landcover_data_to_df(self, df):

        f = data_root + rf'\Base_data\\glc_025\\glc2000_025.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df
    def add_landcover_classfication_to_df(self, df):

        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            landcover=row['landcover_GLC']
            if landcover==0 or landcover==4:
                val_list.append('Evergreen')
            elif landcover==2 or landcover==4 or landcover==5:
                val_list.append('Deciduous')
            elif landcover==6:
                val_list.append('Mixed')
            elif landcover==11 or landcover==12:
                val_list.append('Shrub')
            elif landcover==13 or landcover==14 or landcover==15:
                val_list.append('Grass')
            elif landcover==16 or landcover==17 or landcover==18:
                val_list.append('Cropland')
            else:
                val_list.append(np.nan)
        df['landcover_classfication']=val_list

        return df


        pass
    def add_maxmium_LC_change(self, df): ##

        f = rf'E:\CCI_landcover\trend_analysis_LC\\LC_max.tif'

        array, origin, pixelWidth, pixelHeight, extent = ToRaster().raster2array(f)
        array[array <-99] = np.nan

        LC_dic =DIC_and_TIF().spatial_arr_to_dic(array)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix

            val= LC_dic[pix]
            df.loc[i,'LC_max'] = val
        return df

    def add_aridity_to_df(self,df):  ## here is original aridity index not classification

        f=data_root+rf'Base_data\dryland_AI.tif\\dryland.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(fdir + f)

        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name='Aridity'
        print(f_name)
        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val=val_dic[pix]
            if val<-99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f'{f_name}']=val_list

        return df



    def add_AI_classfication(self, df):

        f = data_root + rf'\\Base_data\dryland_AI.tif\\dryland_classfication.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val==0:
                label='Arid'
            elif val==1:
                label='Semi-Arid'
            elif val==2:
                label='Sub-Humid'
            elif val<-99:
                label=np.nan
            else:
                raise




            val_list.append(label)

        df['AI_classfication'] = val_list
        return df





    def show_field(self, df):
        for col in df.columns:
            print(col)
        return df
        pass


class Check_data:
    def __init__(self):
        pass
    def run(self):
        # self.plot_spatial()
        self.testrobinson()
        pass
    def plot_spatial(self):

        # fdir =  data_root + rf'Monthly_DIC\GLEAM_SMroot\\'
        fdir= rf'D:\Project4\Result\extract_GS\OBS_LAI_extend\\'



        for f in os.listdir(fdir):




            dic=T.load_npy(fdir+f)
            # dic=T.load_npy_dir(f)

                # for f in os.listdir(fdir):
                #     if not f.endswith(('.npy')):
                #         continue
                #
                #     dic_i=T.load_npy(fdir+f)
                #     dic.update(dic_i)

            len_dic={}

            for pix in dic:
                r,c=pix
                # if r<480:
                #     continue
                vals=dic[pix]
                if len(vals)<1:
                    continue
                if np.isnan(np.nanmean(vals)):
                    continue


                # plt.plot(vals)
                # plt.show()


                len_dic[pix]=np.nanmean(vals)
                # len_dic[pix]=np.nanstd(vals)

                # len_dic[pix] = len(vals)
            arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)


            plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmin=-10,vmax=40)
            plt.colorbar()
            plt.title(f)
            plt.show()

    def testrobinson(self):

        fdir = result_root + rf'\monte_carlo_trend_extreme\difference\\'

        # fpath_p_value = result_root+rf'intra_CV\trend_analysis\\CV_LAI4g_p_value.tif'
        temp_root = r'trend_analysis\anomaly\\temp_root\\'
        T.mk_dir(temp_root, force=True)
        # fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        # flag = 0


        for f in os.listdir(fdir):
            # ax=axs.ravel()[flag]

            if not f.endswith('.tif'):
                continue



            fname = f.split('.')[0]
            print(fname)

            m, ret = Plot().plot_Robinson(fdir + f, vmin=-0.3, vmax=0.3, is_discrete=False, colormap_n=5, )

            # Plot().plot_Robinson_significance_scatter(m,fpath_p_value,temp_root,0.05)
            fname= f.split('.')[0]


            plt.title(f'{fname}_(%/year)')
            # plt.title(f'{fname}_(%)')

            plt.show()




def main():
    # Data_preprocessing().run()
    # Calculate_extreme_event().run()
    Monte_carlo_trend_extreme().run()
    # Monte_carlo_CV_extreme().run()
    # Monte_carlo_CV_moderate().run()
    # Machine_learning_cluster().run()

    # Monte_carlo_trend_moderate().run()


    # build_dataframe().run()
    # Probability_density().run()


    # Check_data().run()




    pass

if __name__ == '__main__':
    main()
