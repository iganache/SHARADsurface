# python file to extract surface reflectivity values form SHARAD data
# paralleized

import os
import glob
import argparse
import multiprocessing as mp

import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd

import rasterio as rio
from pyproj import Transformer
from pyproj import CRS

import scipy.signal as signal
import scipy.integrate as integrate

import dem_utils as dem_utils


def cli():
    parser = argparse.ArgumentParser(
        prog="sharad_surface_reflectivity.py",
        description="Computes surface reflectivity from SHARAD radargrams, clutter simulation, and geometry files",
    )
    parser.add_argument("data_dir", type=str, help="path to radargram directory")
    parser.add_argument("geom_dir", type=str, help="path to geom files directory")
    parser.add_argument("clutter_dir", type=str, help="path to clutter simulation directory")
    parser.add_argument("demfile", type=str, help="dem file path")
    parser.add_argument("outfile", type=str, help="output file path")

    return parser.parse_args()


def campbell_rgram_roughness_param(rgrm_data, fl, shift_value = 1000):
# compute roughness parameter defined in campbell et al. 2016
        sur_echo = []
        sur_echo_loc = []
        trace_num = []
        
        for n in range(rgrm_data.shape[1]):
            trace_num.append(n)
            trace = rgrm_data[:,n]
            max_echo = np.max(trace[fl[n]-20:fl[n]+20], axis=0)
            if max_echo ==0.0 or math.isnan(max_echo):
                sur_echo.append(np.nan)
                sur_echo_loc.append(0)
            else:     
                sur_echo.append(max_echo)
                sur_echo_loc.append(np.where(trace==max_echo)[0][0])
            
            
        sur_echo = np.array(sur_echo)
        sur_echo_loc = np.array(sur_echo_loc)
        trace_num = np.array(trace_num)
        
        # roughness parameter estimation
            # 1) shift the radargram
        rows = shift_value
        shifted_rgr = np.zeros((rows, sur_echo_loc.shape[0]), dtype=np.float32)
        for i in range(sur_echo_loc.shape[0]):
            if(sur_echo_loc[i]!=0):
                shifted_rgr[:,i] = rgrm_data[sur_echo_loc[i]-int(rows/2):sur_echo_loc[i]+int(rows/2), i]
            
           # 2) seven-sample boxcar filter (3.22 km along track)
        n = 7
        boxcar = np.ones(n)/n
        shifted_rgr_filt = np.zeros_like(shifted_rgr)
        for i in range(rows):
            shifted_rgr_filt[i] = np.convolve(shifted_rgr[i], boxcar, mode="same")
            
            # 3) Echo in 1st 20 delay bins divided by peak power
        P_peak = shifted_rgr_filt[int(rows/2)]
        P_int = np.sum(shifted_rgr_filt[int(rows/2):int(rows/2)+20], axis = 0)
        roughness_param = P_int/P_peak

        return trace_num, sur_echo_loc, sur_echo, roughness_param

def get_DEM_patch(demfile, minx, maxx, miny, maxy):
    # open MOLA DEM file and extract extent defined by minx, maxx, miny, maxy
    with rio.open(demfile) as ds:
        # find pixel coordinates of window UL and LR (input format: lon, lat; output format: row, col)
        uly, ulx = ds.index(minx, maxy)
        lry, lrx = ds.index(maxx, miny)
        
        # set window height and width
        winH = int(lry-uly) 
        winW = int(lrx - ulx) 

        # create window and extract data within
        win = rio.windows.Window(ulx, uly, winW, winH)
        mola_dem = ds.read(1, window=win)

        # get pixel coords and lat/lon coords meshgrid of window
        dx = ds.transform[0]
        dy = ds.transform[4]

        mola_cols, mola_rows = np.meshgrid(np.arange(ulx, lrx), np.arange(uly, lry))
        mola_lon, mola_lat = rio.transform.xy(ds.transform, mola_rows, mola_cols)
        mola_lon= np.array(mola_lon)
        mola_lat = np.array(mola_lat)
        
      
    ds.close()
    return mola_dem, mola_lat, mola_lon
    
def slope_multiprocess(df, demfile, extent, nproc=4):
    print("beginning MOLA file open and data extraction")
    # get extent of footprint lat/lon for windowing DEM data
    miny = extent[0] - 1
    maxy = extent[1] + 1
    minx = extent[2] - 1
    maxx = extent[3] + 1

    print(extent)

    # convert MOLA lat/lon to MCMF coordinate system
    mola = get_DEM_patch(demfile, minx, maxx, miny, maxy)
    print("end MOLA file open and data extraction")


    # convert geodetic values to MCMF coordinate system
    print("beginning coordinate conversion")
    #subradar_points
    sr_lat = df.latitude.to_numpy()
    sr_lon = df.longitude.to_numpy()
    sr_radius = df.radius_mars.to_numpy()
    sc_height = df.radius_mro.to_numpy() - df.radius_mars.to_numpy()
    coords = list(zip(df.latitude, df.longitude, df.radius_mars))

    args = []
    for i in range(len(coords)):
        args.append([coords[i], sc_height[i], mola])
        
    pool = mp.Pool(processes=nproc)
    
    # # results is returned as a pd series
    
    slope_results = pool.starmap(dem_utils.median_slope, args)
    # print(slope_results)
    # df["FZ_slope_deg"] = slope_results

    return slope_results


def main():
    args = cli()
    data_dir = args.data_dir
    geom_dir = args.geom_dir
    clutter_dir = args.clutter_dir
    demfile = args.demfile
    outfile = args.outfile
    

    # data_dir = "/media/indujaa/Extreme SSD/SHARADdata_tyrrhena_mons/data/rgram"
    # geom_dir = "/media/indujaa/Extreme SSD/SHARADdata_tyrrhena_mons/data/geom"
    # clutter_dir = "/media/indujaa/Extreme SSD/SHARADdata_tyrrhena_mons/cluttersims"
    
    tracks = []
    rgrams = glob.glob(data_dir+"/**/*.img", recursive = True)
    for name in rgrams:
        tracks.append(os.path.basename(name)[0:10])
    geoms = glob.glob(geom_dir+"/**/*.tab", recursive = True)
    sims = glob.glob(clutter_dir+"/**/*.csv", recursive = True)

    dfs = []

    for i, tid in enumerate(tracks):
               
        # reading geom file 
        try:
            geom = glob.glob(geom_dir+"/**/*"+tid+"*.tab", recursive = True)[0]
            geomCols = ["trace", "time","lat","lon","R_mars","R_mro","radiVel","tangVel","SZA","phaseD",]
            geomdf = pd.read_csv(geom, names=geomCols, index_col=False)
            sur_lat = geomdf.lat.to_numpy()
            sur_lon = geomdf.lon.to_numpy()
            rad_mars = geomdf.R_mars.to_numpy()
            rad_mro = geomdf.R_mro.to_numpy()
            
        except:
            print("geom file missing")
            break
            
            
        # reading clutter simulation file 
        try:
            sim = glob.glob(clutter_dir+"/**/*"+tid+"*.csv", recursive = True)[0]
            fl = pd.read_csv(sim)["FirstLine"].to_numpy()
        except:
            print("clutter file missing")
            break
            
        
        # reading radargram file and extracting max surface return amplitude
        rgrm = glob.glob(data_dir+"/**/*"+tid+"*.img", recursive = True)[0]
    #     print(tid, rgrm, sim)
        nsamples = 3600
        rgrm_data = np.reshape(np.fromfile(rgrm, dtype=np.float32), (nsamples, -1))

        trace_num, sur_echo_loc, sur_echo, roughness_param = campbell_rgram_roughness_param(rgrm_data, fl)       
        dfs.append(pd.DataFrame({'track_id': tid,
                                'trace': trace_num,
                                'sample_sur': sur_echo_loc,
                                'latitude': sur_lat, 
                                'longitude': sur_lon,
                                'radius_mars': rad_mars, 
                                'radius_mro': rad_mro,
                                'surface_echo': sur_echo,
                                'roughness_param': roughness_param}))

    # combine dataframe
    final_df=pd.concat(dfs, ignore_index=True) 
    final_df["surface_echo_corr"] = final_df["surface_echo"] * final_df["roughness_param"]**2
    final_df.to_csv(outfile, index = False)
    

    # # total spatial extent of the dataframe
    # miny = final_df.latitude.min() 
    # maxy = final_df.latitude.max() 
    # minx = final_df.longitude.min()
    # maxx = final_df.longitude.max()
    # extent = [miny, maxy, minx, maxx]
    

    # # final_df = final_df[final_df["track_id"] == "s_07659501"]
    # final_df = final_df.iloc[156865:156969]

    # FZ_slope = slope_multiprocess(final_df, demfile, extent, nproc=24)
    # FZ_slope_arr = np.array(FZ_slope)

    # final_df["FZ_slope_median_deg"] = FZ_slope_arr[:,0]
    # final_df["FZ_slope_mean_deg"] = FZ_slope_arr[:,1]
    # final_df["FZ_slope_stdev_deg"] = FZ_slope_arr[:,2]

    # final_df.to_csv(outfile, index = False)


main()