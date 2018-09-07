import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import glob
import cartopy.crs as ccrs
import scipy.interpolate
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
mycmap = discrete_cmap(21,'spectral')
divcmap = discrete_cmap(21,'RdBu')
#dates = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14', '15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
dates = ['01_06','01_12','01_18','02_00','02_06','02_12','02_18','03_00','03_06','03_12','03_18','04_00','04_06','04_12','04_18','05_00','05_06','05_12','05_18','06_00','06_06','06_12','06_18','07_00','07_06','07_12','07_18','08_00','08_06','08_12','08_18']
date_time = np.zeros([len(dates)])
for i,date in enumerate(dates):
    date_time[i]=float(date[0:2])+float(date[3:])/24	
colmean_L132 = np.zeros([len(dates)])
colmean_L72 = np.zeros([len(dates)])
plt.figure(figsize=[50,10])
def area(res):
#calculate the area of each lat/lon grid:
	if res=='c90':
		c=0
	else:
		c=1
	latit = np.linspace(-90,90,num  = 181+c*180)
	longit = np.linspace(-180,180,num = 360+c*360)
	Area = np.ones([181+c*180,360+c*360])
	for tick_lat in range(len(latit)):
	    for tick_lon in range(len(longit)):
		latit_low = (latit[tick_lat]-0.5/(c+1))*np.pi/180
		latit_high = (latit[tick_lat]+0.5/(c+1))*np.pi/180
		longit_low = longit[tick_lon]-0.5/(c+1)
		longit_high = longit[tick_lon]+0.5/(c+1)
	       #square meters:
		Area[tick_lat,tick_lon] = np.pi/180*6378.1e3**2*abs(np.sin(latit_high)-np.sin(latit_low))*abs(longit_high-longit_low)#dim: 181,360
	return(Area)
plumes = np.arange(1,7,1).astype(str)
def tracermass(dataset,res,plume):
	mass=np.sum(dataset['PASV'+plume+'dry'].values*dataset['GCC_AIRNUMDEN'].values*dataset['GCC_AIRVOL'].values*10**15)
	return(mass)
colors=['k','b']
res = 'c90'
col='k'
for i,plume in enumerate(plumes,1):
	for tick_d in range(len(dates)):
		if res=='c90':
			filenames132 = glob.glob('../c90_RnPbBe_L132/holding_plume13/geosgcm_gcc_lev/*201303'+dates[tick_d]+'*z.nc4')[0]
			filenames72 = glob.glob('../c90_RnPbBe_L72/holding_plume13/geosgcm_gcc_lev/*.201303'+dates[tick_d]+'*z.nc4')[0]
		else:
			filenames132 = glob.glob('../c180_L132/holding_plume13/geosgcm_gcc_lev/*201303'+dates[tick_d]+'*z.nc4')[0]
			filenames72 = glob.glob('../c180_L72/holding_plume13/geosgcm_gcc_lev/*.201303'+dates[tick_d]+'*z.nc4')[0]
		print filenames132
		print filenames72
		lev132 = np.zeros([132])
		lev72 = np.zeros([72])
		print dates[tick_d]
		acccolmean_L132 = np.zeros([1])
		acccolmean_L72 = np.zeros([1])
		ds = xr.open_dataset(filenames132)
		dr = xr.open_dataset(filenames72)
		colmean_L132[tick_d] = tracermass(ds,res,plume)
		colmean_L72[tick_d] = tracermass(dr,res,plume)
	ax=plt.subplot(1,len(plumes),i)
	L132, = ax.plot(date_time, colmean_L132,color=col,linestyle='--',linewidth=3,markersize =4, label = res+'_'+'L132')
	L72, = ax.plot(date_time, colmean_L72,color=col,linestyle=':',linewidth=3,label =res+'_'+'L72',markersize = 4)
	
	plt.legend(handles=[L132,L72],loc = 'best',frameon=False)
	plt.xlabel('day')
	plt.ylabel('mol tracer')
	plt.setp(ax.spines.values(), linewidth=3)
plt.suptitle('c90 (black) vs. c180 (blue)',fontsize=14,fontweight='bold')
plt.savefig('plume_images_hres/constrac_c90vc180.png')
