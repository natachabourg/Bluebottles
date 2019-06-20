import netCDF4
ncfile = netCDF4.Dataset('extrapolated_u_v.nc', mode='w', format='NETCDF4_CLASSIC')
print(ncfile)

lat_dim = ncfile.createDimension('lat', 12000) #can be appended
lon_dim = ncfile.createDimension('lon', 12000) #can be appended
time_dim = ncfile.createDimension('time', 111) #can be appended
for dim in ncfile.dimensions.items():
    print(dim)
    
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'

time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'hours since 1800-01-01'
time.long_name = 'time'
u_cur = ncfile.createVariable('u_cur', np.float64, ('time','lat','lon'))
u_cur.units = 'm/s'
u_cur.standard_name = 'eastward_current_velocity'
v_cur = ncfile.createVariable('v_cur', np.float64, ('time','lat','lon'))
v_cur.units = 'm/s'
v_cur.standard_name = 'northward_current_velocity'