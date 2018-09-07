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
#dates = ['01_06','01_12','01_18','02_00','02_06','02_12','02_18','03_00','03_06','03_12','03_18','04_00','04_06','04_12','04_18','05_00','05_06','05_12','05_18','06_00','06_06','06_12','06_18','07_00','07_06','07_12','07_18','08_00','08_06','08_12','08_18']
dates = ['01_06','01_12','01_18','02_00','02_06','02_12','02_18','03_00','03_06','03_12','03_18','04_00','04_06','04_12','04_18','05_00','05_06','05_12','05_18','06_00','06_06','06_12','06_18','07_00']
#dates = ['06','07','08','09','10','11','12']
date_time = np.zeros([len(dates)])
for i,date in enumerate(dates):
    date_time[i]=float(date[0:2])+float(date[3:])/24	
colmean_L132 = np.zeros([len(dates)])
colmean_L72 = np.zeros([len(dates)])

#calculate the area of each lat/lon grid:
#latit = np.linspace(-90,90,num  = 181)
#longit = np.linspace(-180,180,num = 360)
#Area = np.ones([181,360])
#for tick_lat in range(len(latit)):
#    for tick_lon in range(len(longit)):
#        latit_low = (latit[tick_lat]-0.5)*np.pi/180
#        latit_high = (latit[tick_lat]+0.5)*np.pi/180
#        longit_low = longit[tick_lon]-0.5
#        longit_high = longit[tick_lon]+0.5
        #square meters:
#        Area[tick_lat,tick_lon] = np.pi/180*6378.1e3**2*abs(np.sin(latit_high)-np.sin(latit_low))*abs(longit_high-longit_low)#dim: 181,360
def entropy(ZL_da, PL_da, T_da, mr_da,vertres):
	#zedge = 0.5*(ZL_da.values[0,np.arange(0,vertres,2),:,:]+ZL_da.values[0,np.arange(1,vertres+1,2),:,:])#dim: 71,181,360
	zedge = (ZL_da.values[0,1:,:,:] + ZL_da.values[0,:-1,:,:])/2
	Area3d = np.repeat(Area[np.newaxis,:,:], vertres-2,axis=0)
	dz = zedge[np.arange(0,vertres-2,1),:,:]-zedge[np.arange(1,vertres-1,1),:,:] #dim: 70,181,360
	dv = Area3d * np.squeeze(dz)
	air_mass = dv * 100 * PL_da.values[0,np.arange(1,vertres-1,1),:,:] /8.314/T_da.values[0,np.arange(1,vertres-1,1),:,:]*29
	mixingrat = mr_da.values[0,np.arange(1,vertres-1,1),:,:]*29
	print np.log(mixingrat*29, where=mixingrat>0 )
	entropy = np.sum(air_mass*mixingrat*np.log(mixingrat*29, where=mixingrat>0),axis = None)
	return(entropy)
def easyent(airmass_da, mr_da):
	mixing_ratio=mr_da.values * 29
	entropy = np.sum(airmass_da.values*mixing_ratio*np.log(mixing_ratio, where=mixing_ratio>0),axis=None)
	return(entropy)
#def maxvmr(mr_da):
#	mr_roll = np.roll(mr_da.values,180,axis=3)
#	mr_range = mr_roll[0,:,:,150:230]
#	long = np.roll(np.arange(-180,181,0.5),361,axis=0)
#	#if tick_f==0:
#		print 'beg'+ str(long[300+tick_d*20])
#		print 'end'+ str(long[460+tick_d*4])
#	mr_max = np.max(mr_range)
#	return(mr_max)
#colors=['k','b']
ress=['c90']
colors = ['k','b','y','r','m','c']
tracers = map(str,np.arange(1,7,1))
for color,tracer in zip(colors,tracers):
	for res in ress:
		for tick_d in range(len(dates)):
			if res=='c90':
				filenames132 = glob.glob('../c90_RnPbBe_L132/scratch/*201303'+dates[tick_d]+'*z.nc4')
				filenames72 = glob.glob('../c90_RnPbBe_L72/scratch/*.201303'+dates[tick_d]+'*z.nc4')
			else:
				filenames132 = glob.glob('../c180_L132/scratch/*.201303'+dates[tick_d]+'*z.nc4')
				filenames72 = glob.glob('../c180_L72/scratch/*.201303'+dates[tick_d]+'*z.nc4')
				#filenames132 = glob.glob('../c180_L132/holding/geosgcm_gcc_lev/*201303'+dates[tick_d]+'*z.nc4')
				#filenames72 = glob.glob('../c180_L72/holding/geosgcm_gcc_lev/*.201303'+dates[tick_d]+'*z.nc4')
			print filenames132
			print filenames72
			num_tick_f = np.min([len(filenames72),len(filenames132)])
			lev132 = np.zeros([132])
			lev72 = np.zeros([72])
			print dates[tick_d]
			print "number of working files:" + str(num_tick_f)
			acccolmean_L132 = np.zeros([1])
			acccolmean_L72 = np.zeros([1])
			for tick_f in range(num_tick_f):
				ds = xr.open_dataset(filenames132[tick_f])
				dr = xr.open_dataset(filenames72[tick_f])
				#lev72 = np.reshape(dr['PL'].mean(dim=['lat','lon']).values,[72])
				#lev132 = np.reshape(ds['PL'].mean(dim=['lat','lon']).values,[132])
		#		if tick_d == 0 and tick_f == 0:
		#			k_132 = entropy(ds['ZL'],ds['PL'],ds['T'],ds['TRC_PASV'],132)
		#			k_72 = entropy(dr['ZL'],dr['PL'],dr['T'],dr['TRC_PASV'],72)
		#			entropy_L132 = 1
		#			entropy_L72 = 1
				#entropy_L132 =entropy(ds['ZL'],ds['PL'],ds['T'],ds['TRC_PASV'],132)
				#entropy_L72 =entropy(dr['ZL'],dr['PL'],dr['T'],dr['TRC_PASV'],72)
				#entropy_L132 = easyent(ds['AIRMASS'],ds['TRC_PASV'])	
				#entropy_L72 = easyent(dr['AIRMASS'],dr['TRC_PASV'])
				maxvmr_L132 = np.max(ds['TRC_PASV'+tracer])*29/48
				maxvmr_L72 = np.max(dr['TRC_PASV'+tracer])*29/48
				print 'max val'+str(maxvmr_L132)
				acccolmean_L132 = maxvmr_L132
				acccolmean_L72 = maxvmr_L72
		#		acccolmean_L132 += np.squeeze(ds['TRC_PASV'].sum(dim = 'lat').values)
		#		acccolmean_L72 += np.squeeze(dr['TRC_PASV'].sum(dim = 'lat').values)		
			#lon = np.linspace(-180,180,num = 360)
			print np.shape(acccolmean_L132)
			if tick_d==0:
				k132=acccolmean_L132
				k72 = acccolmean_L72
				print 'L132 max = '+ str(k132)
				print 'L72 max = ' + str(k72)
			colmean_L132[tick_d] = acccolmean_L132/2e-7
			colmean_L72[tick_d] = acccolmean_L72/2e-7
			print colmean_L132
			print colmean_L72
			
			#interpolate 72 layers to 132 layers
			#interp = scipy.interpolate.interp2d(lon,lev72,colmean_L72_lowres,kind = 'linear')
			#colmean_L72 = interp(lon,lev132)

			# L72 plot
		#ax = plt.subplot(14,3,1+3*tick_d)
			#pc = ax.pcolormesh(lon,lev132,colmean_L72,vmin=0,vmax=20)
			#plt.xlabel('longitude')
			#plt.xlim([-180,180])
			#plt.gca().invert_yaxis()
			#plt.title('L72 day'+dates[tick_d])
			#plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average mixing ratio kg/kg')

			# L132 plot
			#ax1 =plt.subplot(14,3,2+3*tick_d)
			#plt.xlabel('longitude')
			#plt.xlim([-180,180])
			#plt.title('L132 day'+dates[tick_d])
			#pc = ax1.pcolormesh(lon,lev132,colmean_L132,vmin = 0,vmax =20)
			#plt.gca().invert_yaxis()
			#plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average mixing ratio kg/kg')

			# L72 v L132 plot
			#perc_avg = np.divide((colmean_L72-colmean_L132),colmean_L72,out=np.zeros_like(colmean_L72-colmean_L132),where=colmean_L72!=0)*100
			#ax2 =plt.subplot(14,3,3+3*tick_d)
			#plt.xlabel('longitude')
			#plt.xlim([-180,180])
			#plt.title('day'+dates[tick_d])
			#pc = ax2.pcolormesh(lon,lev132,perc_avg,vmin = -100,vmax =100,cmap = divcmap)
			#plt.gca().invert_yaxis()
			#plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'red:more L132   percent average   blue:more L72')
		ax=plt.subplot(111)
		L132 = plt.plot(np.insert(date_time,0,1), np.insert(colmean_L132,0,1),color=color,linestyle='--',linewidth=3,markersize =4, label = res+'_'+'L132')
		plt.hold(True)
		plt.xlabel('day')
		plt.ylabel('fraction of max VMR')
		L72 = plt.plot(np.insert(date_time,0,1), np.insert(colmean_L72,0,1),color=color,linestyle=':',linewidth=3,label =res+'_'+'L72',markersize = 4)
		if tracer=='1':
			plt.legend(loc = 'best',frameon=False)
plt.setp(ax.spines.values(), linewidth=3)
plt.title('plume height versus VMR',fontsize=14,fontweight='bold')
plt.savefig('plume_images_hres/maxvmr_c90_12.png')
	
