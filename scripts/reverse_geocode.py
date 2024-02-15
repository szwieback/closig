'''
Created on Feb 8, 2024

@author: simon
'''
import numpy as np
from pathlib import Path

def read_latlon(p0, boxcar_thin=None, crop=None):
    fnlat = p0 / 'lat.rdr.full'
    fnlon = p0 / 'lon.rdr.full'
    fnxml = p0 / 'lat.rdr.full.aux.xml'
    with open(fnxml, 'r') as f:
        s = f.readlines()
        l = [l.strip() for l in s if '<MDI key="samples">' in l][0]
        samples = int(l[l.index('>') + 1:l.index('<', 2)])

    latlon = np.stack(
        (np.fromfile(fnlat, dtype=np.double), np.fromfile(fnlon, dtype=np.double)), axis=0).reshape(2, -1, samples)
    if boxcar_thin is not None:
        ds, dl = boxcar_thin
        latlon = latlon[:, ds // 2::ds, dl // 2::dl]
    if crop is not None:
        latlon = crop(latlon)
    return latlon

def reverse_geocode_landsat(latlon, pls, lsname, fnfinal, bands=(6, 5, 4), overwrite=False):
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform, Resampling    
    if not fnfinal.exists() or overwrite:
        for jb, b in enumerate(bands):
            fnb = pls / (lsname + f'_SR_B{b}.TIF')
            fntmp = pls / (lsname + f'_B{b}.tmp.tif')
            with rasterio.open(fnb) as src:
                im = src.read(1)
                dst_crs = 'EPSG:4326'
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
                with rasterio.open(fntmp, 'w', **kwargs) as dst:
                    reproject(
                            source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                            src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                            dst_crs=dst_crs, resampling=Resampling.nearest)
            with rasterio.open(fntmp) as src:
                im = src.read(1)
                transform = src.transform
            if jb == 0:
                imr = np.zeros(latlon.shape[1:] + (len(bands),), dtype=im.dtype)
            invtransform = ~transform
            for j1 in range(imr.shape[0]):
                for j2 in range(imr.shape[1]):
                    rc = invtransform * (latlon[1, j1, j2], latlon[0, j1, j2])
                    try:
                        imr[j1, j2, jb] = im[round(rc[1]), round(rc[0])]
                    except:
                        pass
        np.save(fnfinal, imr)
    else:
        imr = np.load(fnfinal)
        
def reverse_geocode_Colorado():
    pls = Path('/home/simon/Work/closig/optical/Colorado')
    lsname = 'LC08_L2SP_034032_20200702_20200913_02_T1'
    p0 = Path('/home/simon/Work/closig/stacks/Colorado/')
    fnfinal = pls / (lsname + f'_resampled.npy')
    boxcar_thin = (9, 25)
    crop = lambda im: np.moveaxis(im[..., 80:, 80:][...,:376,:895], 1, 2)#im[10:-10, 40:-40, ...]
    latlon = read_latlon(p0, boxcar_thin=boxcar_thin, crop=crop)
    print(latlon[:, 0, 0])
    print(latlon[:, 0, -1])
    reverse_geocode_landsat(latlon, pls, lsname, fnfinal, overwrite=False)

def reverse_geocode_NewMexico():
    pls = Path('/home/simon/Work/closig/optical/NewMexico')
    lsnames = ('LC09_L2SP_032036_20220515_20230416_02_T1', 'LC08_L2SP_033036_20220514_20220519_02_T1')

    for lsname in lsnames:
        p0 = Path('/home/simon/Work/closig/stacks/NewMexico/')
        fnfinal = pls / (lsname + f'_resampled.npy')
        boxcar_thin = (9, 25)
        crop = lambda im: np.moveaxis(im[..., 10:, 90:][...,:444,:905], 1, 2)
        latlon = read_latlon(p0, boxcar_thin=boxcar_thin, crop=crop)
        reverse_geocode_landsat(latlon, pls, lsname, fnfinal, overwrite=False)
    ims = []
    for lsname in lsnames:
        fnfinal = pls / (lsname + f'_resampled.npy')
        im = np.load(fnfinal)
        invalid = (im == 0)
        im = im.astype(np.float32)
        im[invalid] = np.nan
        ims.append(im)
    im = np.stack(ims, axis=0)
    im = np.nanmean(im, axis=0)
    np.save(pls / 'combined_resampled.npy', im)
if __name__ == '__main__':
    # reverse_geocode_Colorado()
    reverse_geocode_NewMexico()
