import xarray as xr
import numpy as np
import os
import glob
res = float(raw_input('what is your vertical resolution?: '))
filepath=['/discover/nobackup/mhovdest/initial_conditions/132_lvl_initial_GEOSChem_rst.2x25_RnPbBePasv_72L.edit_levcoord.nc','/discover/nobackup/mhovdest/initial_conditions/initial_GEOSChem_rst.2x25_RnPbBePasv_72L.edit_levcoord.nc']
tick_f=0
if res==72:
	tick_f=1
plumes = int(raw_input('how many plumes do you want to initiate?: '))
tracers=map(str,np.arange(1,plumes+1,1))
path, filename = os.path.split(filepath[tick_f])
dataset = xr.open_dataset(filepath[tick_f])
lon_b = float(raw_input('initial longitude?: '))
lon_e = float(raw_input('final longitude?: '))
lat_b = float(raw_input('initial latitude?: '))
lat_e = float(raw_input('final latitude?: '))
const_conc = float(raw_input('what constant concentration do you want in the plume (v/v)?: '))
ds = xr.Dataset()
ds = dataset
dimen = ds['SPC_PASV'].values.shape
vals= np.zeros(dimen)
for tracer in zip(tracers):
        lev_b = 0
	lev_e = 0
	name = 'SPC_PASV' + tracer[0]
	print 'name of tracer: ' + name
	begin = int(raw_input('bottom level (72/132 is near surface):'))
	end = int(raw_input('top level (1 is top of atmosphere):'))
	lev_b = int(res+1)-begin
	lev_e = int(res+1)-end
	print "diagnostic of levels: "
	print lev_b
	print lev_e
	ds[name]=xr.full_like(ds.SPC_PASV,fill_value=0)
	ds[name].encoding = ds.SPC_PASV.encoding
	ds[name].loc[dict(lon=slice(lon_b,lon_e),lat=slice(lat_b,lat_e),lev=slice(lev_b,lev_e))]=const_conc
	print ds[name]
	print "----------------------------"
ds = ds.drop('SPC_PASV')
print ds
ds.to_netcdf('/discover/nobackup/mhovdest/initial_conditions/PASV_plume_'+filename)
