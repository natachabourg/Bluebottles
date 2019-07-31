S = shaperead('NSW_STATE_POLYGON_shp.shp')
figure
for p=1:319
    hold on
    lon=S(p).X;
    lat=S(p).Y;
   % plot(lon,lat,'g','linewidth',3)
    hold on
    p
    pause(1)
end

% the main boundary of NSW is
p=294;
lon_nsw=S(294).X;
lat_nsw=S(294).Y;
save('NSW_boundary','lon_nsw','lat_nsw')
