# compute median slope at nadir

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from pyproj import Transformer
from pyproj import CRS



def geodetic2mcmf(lon, lat, radius):
# convert subradar points to MCMF coordinate system
    MCMF_WKT = 'GEODCRS["Mars (2015) Cartesian", \
    DATUM["Mars (2015)", \
        ELLIPSOID["Mars (2015)",3396190,169.894447223612, \
            LENGTHUNIT["metre",1]], \
        ANCHOR["Viking 1 lander: 47.95137 W"]], \
    PRIMEM["Reference Meridian",0, \
        ANGLEUNIT["degree",0.0174532925199433]], \
    CS[Cartesian,3], \
        AXIS["(X)",geocentricX, \
            ORDER[1], \
            LENGTHUNIT["metre",1, \
                ID["EPSG",9001]]], \
        AXIS["(Y)",geocentricY, \
            ORDER[2], \
            LENGTHUNIT["metre",1, \
                ID["EPSG",9001]]], \
        AXIS["(Z)",geocentricZ, \
            ORDER[3], \
            LENGTHUNIT["metre",1, \
                ID["EPSG",9001]]]]"'

    marsGeoCRS_WKT = 'GEOGCS["Mars (2015) - Sphere / Ocentric", \
                        DATUM["Mars (2015) - Sphere", \
                        SPHEROID["Mars (2015) - Sphere",3396190,0, AUTHORITY["IAU","49900"]], \
                        AUTHORITY["IAU","49900"]], \
                        PRIMEM["Reference Meridian",0,AUTHORITY["IAU","49900"]], \
                        UNIT["degree",0.0174532925199433, AUTHORITY["EPSG","9122"]], AUTHORITY["IAU","49900"]]'

    latlon2mcmf = Transformer.from_crs(CRS(marsGeoCRS_WKT ), CRS(MCMF_WKT))
    xMCMF, yMCMF, zMCMF = latlon2mcmf.transform(lon, lat, radius)
    return xMCMF, yMCMF, zMCMF


def MCMF2ENU(lon, lat, srCoords, mCoords):
# coordinate transformation to local tangential plane
    srX, srY, srZ = srCoords
    mX, mY, mZ = mCoords
    rtMatrix = np.array(
                        [[-np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)), 0],
                         [-np.sin(np.deg2rad(lat))*np.cos(np.deg2rad(lon)), -np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lat))],
                         [np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon)), np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon)), np.sin(np.deg2rad(lat))]]
                        )
    trans = np.array([mX.flatten()-srX, mY.flatten()-srY, mZ.flatten()-srZ])
#     print(rtMatrix.shape, trans.shape)
    x,y,z = np.matmul(rtMatrix, trans)
    return x,y,z



def median_slope(sr_coords, sc_height, dem):
    sr_lat = sr_coords[0]
    sr_lon = sr_coords[1]
    sr_radius = sr_coords[2]

    mola_dem = dem[0]
    mola_lat = dem[1]
    mola_lon = dem[2]

    # print(sr_lon, sr_lat)    
    # fig, ax = plt.subplots(1,1)
    # ax.imshow(mola_dem)
    # plt.show()

    #coordinate conversion
    sr_xMCMF, sr_yMCMF, sr_zMCMF = geodetic2mcmf(sr_lon, sr_lat, sr_radius*1e3-3396190)
    mola_xMCMF, mola_yMCMF, mola_zMCMF = geodetic2mcmf(mola_lon.flatten(), mola_lat.flatten(), mola_dem.flatten())
    mola_xMCMF = mola_xMCMF.reshape(mola_lon.shape)
    mola_yMCMF = mola_yMCMF.reshape(mola_lon.shape)
    mola_zMCMF = mola_zMCMF.reshape(mola_lon.shape)


    # recenter MOLA
    mola_xENU,mola_yENU,mola_zENU = MCMF2ENU(sr_lon, sr_lat, [sr_xMCMF, sr_yMCMF, sr_zMCMF], [mola_xMCMF, mola_yMCMF, mola_zMCMF])
    mola_xENU = mola_xENU.reshape(mola_lon.shape)
    mola_yENU = mola_yENU.reshape(mola_lon.shape)
    mola_zENU = mola_zENU.reshape(mola_lon.shape)


    #determine cells within Fresnel Zone
    fzRad = np.sqrt(15*sc_height*1e3/2)

    # inFZ =  ( (mola_xMCMF-sr_xMCMF)**2  + (mola_yMCMF-sr_yMCMF)**2) / fzRad**2  <=1 
    inFZ =  ( (mola_xENU)**2  + (mola_yENU)**2) / fzRad**2  <=1 


    # Plotting for debugging
    # ( (mola_zMCMF-sr_zMCMF)**2/ fzRadX**2 ) <= 1
    # print(np.count_nonzero(inFZ))

    # fz_xMCMF = mola_xENU.copy()
    # fz_yMCMF = mola_yENU.copy()
    # fz_zMCMF = mola_zENU.copy()
    
    # fz_xMCMF[inFZ==False] = np.nan
    # fz_yMCMF[inFZ==False] = np.nan
    # fz_zMCMF[inFZ==False] = np.nan


    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(fz_xMCMF, aspect="equal")
    # ax[1].imshow(fz_yMCMF, aspect="equal")
    # ax[2].imshow(fz_zMCMF, aspect="equal")
    # plt.show()
    

    # caluclate slope
    # print("beginning MOLA slope calculation")
    # create empty arrays
    mx = np.zeros_like(mola_xENU)
    my = np.zeros_like(mola_xENU)
    mtotal = np.zeros_like(mola_xENU)
    

    # calculate slope in x
    # backward difference for the 1st column
    mx[:,0] = (mola_zENU[:,1] - mola_zENU[:,0]) / np.sqrt((mola_xENU[:,1] - mola_xENU[:,0])**2 + (mola_yENU[:,1] - mola_yENU[:,0])**2)
    # forward difference for the last column
    mx[:,-1] = (mola_zENU[:,-1] - mola_zENU[:,-2]) / np.sqrt((mola_xENU[:,-1] - mola_xENU[:,-2])**2 + (mola_yENU[:,-1] - mola_yENU[:,-2])**2)  
    # central difference elsewhere
    mx[:, 1:-1] = ((mola_zENU[:,2:] - mola_zENU[:,1:-1]) / np.sqrt((mola_xENU[:,2:] - mola_xENU[:,1:-1])**2 + (mola_yENU[:,2:] - mola_yENU[:,1:-1])**2)) - \
                   ((mola_zENU[:,1:-1] - mola_zENU[:,:-2]) / np.sqrt((mola_xENU[:,1:-1] - mola_xENU[:,:-2])**2 + (mola_yENU[:,1:-1] - mola_yENU[:,:-2])**2))  
    
    
    # calculate slope in y
    # backward difference for the 1st row
    my[0,:] = (mola_zENU[1,:] - mola_zENU[0,:]) / np.sqrt((mola_xENU[1,:] - mola_xENU[0,:])**2 + (mola_yENU[1,:] - mola_yENU[0,:])**2)
    # forward difference for the last row
    my[-1,:] = (mola_zENU[-1,:] - mola_zENU[-2,:]) / np.sqrt((mola_xENU[-1,:] - mola_xENU[-2,:])**2 + (mola_yENU[-1,:] - mola_yENU[-2, :])**2)  
    # central difference elsewhere
    my[1:-1,:] = ((mola_zENU[2:,:] - mola_zENU[1:-1,:]) / np.sqrt((mola_xENU[2:,:] - mola_xENU[1:-1,:])**2 + (mola_yENU[2:,:] - mola_yENU[1:-1,:])**2)) - \
                   ((mola_zENU[1:-1,:] - mola_zENU[:-2,:]) / np.sqrt((mola_xENU[1:-1,:] - mola_xENU[:-2,:])**2 + (mola_yENU[1:-1,:] - mola_yENU[:-2,:])**2))  
    
    # total slope
    mtotal = np.sqrt(my**2 + mx**2) 
    slope_deg = np.rad2deg(np.arctan(mtotal))
    slope_deg_fz = slope_deg.copy()
    slope_deg_fz[inFZ==False] = np.nan

    median_slope_deg = np.rad2deg(np.arctan(np.median(mtotal[np.where(inFZ == True)])))
    mean_slope_deg = np.rad2deg(np.arctan(np.mean(mtotal[np.where(inFZ == True)])))
    std_slope_deg = np.rad2deg(np.arctan(np.std(mtotal[np.where(inFZ == True)])))

    return [median_slope_deg, mean_slope_deg, std_slope_deg]





    

    