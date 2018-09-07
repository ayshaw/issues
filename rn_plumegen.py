import xarray as xr
import numpy as np
import os
import glob
res = float(raw_input('what is your vertical resolution?: '))
filepath=['/discover/nobackup/mhovdest/initial_conditions/PASV_plume_132_lvl_initial_GEOSChem_rst.2x25_RnPbBePasv_72L.edit_levcoord.nc','/discover/nobackup/mhovdest/initial_conditions/PASV_plume_initial_GEOSChem_rst.2x25_RnPbBePasv_72L.edit_levcoord.nc']
tick_f=0
if res==72:
	tick_f=1
dataset = xr.open_dataset(filepath[tick_f])
lon_b = float(raw_input('initial longitude?: '))
lon_e = float(raw_input('final longitude?: '))
lat_b = float(raw_input('initial latitude?: '))
lat_e = float(raw_input('final latitude?: '))
lev_b = float(res+1)-float(raw_input('bottom level (72/132 is near surface):'))
lev_e = float(res+1)-float(raw_input('top level (1 is top of atmosphere):'))
const_conc = float(raw_input('what constant concentration do you want in the plume (v/v)?: '))
path, filename = os.path.split(filepath[tick_f])
ds = dataset
ds['SPC_Rn'] *= 0
ds['SPC_Rn'].loc[dict(lon=slice(lon_b,lon_e),lat=slice(lat_b,lat_e),lev=slice(lev_b,lev_e))]=const_conc
ds.to_netcdf('/discover/nobackup/mhovdest/initial_conditions/rn_test/'+filename)
