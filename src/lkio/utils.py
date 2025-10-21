import numpy as np

def load_cosmicray_extension(hdu):
    """Function to load the cosmic ray extension into a datacube

    Parameters:
    -----------
    tpf: lk.TargetPixelFile
        Input Target Pixel File

    Returns:
    -------
    cr: np.ndarray
        Array containing cosmic ray fluxes of shape tpf.shape
    """
    cadenceno = hdu[1].data['CADENCENO']

    # Get cosmic ray information
    l = np.where([h.name == 'TARGET COSMIC RAY' for h in hdu])[0]
    if len(l) != 1:
        return None
    h = hdu[l[0]]
    c, x, y, f = [h.data[attr].copy() for attr in ['CADENCENO', 'RAWX', 'RAWY', 'COSMIC_RAY']]
    if len(c) == 0:
        return None
    x -= hdu[1].header['1CRV5P'] #tpf.column
    y -= hdu[1].header['2CRV5P'] #tpf.row

    # Mask down to only cosmic rays in cadences within the TPF
    k = np.in1d(c, cadenceno)
    c, x, y, f = c[k], x[k], y[k], f[k]

    # Map cosmic ray cadences to array indices
    r = {cadenceno[idx]:idx for idx in range(hdu[1].data['FLUX'].shape[0])}
    inv = np.asarray([r[c1] for c1 in c])

    # Build a cube
    cr = np.zeros(hdu[1].data['FLUX'].shape)
    cr[inv, y, x] = f
    return cr

def return_unit(colname):
    #TODO: I think this will move to lkv3, not required for io. 
    for colname in tab.colnames:
        # Ensure units have the correct astropy format
        # Speed-up: comparing units by their string representation is 1000x
        # faster than performing full-blown unit comparison
        unitstr = str(tab[colname].unit)
        if unitstr == "e-/s":
            tab[colname].unit = "electron/s"
        # Fixes issue #1504
        if unitstr == "electron / s": #lightcurve.to_fits() saves as this string
            tab[colname].unit = "electron/s"
        elif unitstr == "pixels":
            tab[colname].unit = "pixel"
        elif unitstr == "ppm" and repr(tab[colname].unit).startswith("Unrecognized"):
            # Workaround for issue #956
            tab[colname].unit = ppm
        elif unitstr == "ADU":
            tab[colname].unit = "adu"
        elif unitstr.lower() == "unitless":
            tab[colname].unit = ""
        elif unitstr.lower() == "degcelcius":
            # CDIPS has non-astropy units
            tab[colname].unit = "deg_C"
        # Rename columns to lowercase
        tab.rename_column(colname, colname.lower())