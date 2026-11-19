import fitsio
import sys, os
#parent_dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('lkdata'))))
sys.path.append('/Users/nthom/TESS_GI/gitNschanch/tessgi/lkdata/src/')
#sys.path.append(parent_dir_name + "/src")
from lkdata import DataSeries,BoolSeries, BitwiseSeries, DataSet

#HDU1
#spoc_hdu
file = "/Users/nthom/.lightkurve/cache/mastDownload/TESS/tess2020324010417-s0032-0000000685125879-0200-s/tess2020324010417-s0032-0000000685125879-0200-s_lc.fits"

# Will I need to add extension to this? 
spoc_data = [('SAP_FLUX', 'SAP_FLUX_ERR'),
                ('SAP_BKG', 'SAP_BKG_ERR'),
                ('PDCSAP_FLUX','PDCSAP_FLUX_ERR'),
                ('PSF_CENTR1','PSF_CENTR1_ERR'),
                ('PSF_CENTR2','PSF_CENTR2_ERR'),
                ('MOM_CENTR1','MOM_CENTR1_ERR'),
                ('MOM_CENTR2','MOM_CENTR2_ERR'),
                ('POS_CORR1', None),
                ('POS_CORR2', None),
                ]

spoc_bitwise = ['QUALITY'] #Bitwise data can't have uncertainties

spoc_bool = []

'''class lkio(object):
    def __init__(self,
                 file,
                 time_key = 'TIME',
                 flux_key = 'PDCSAP_FLUX',
                 flux_err_key = 'PDCSAP_FLUX_ERR',
                 quality = 'QUALITY'):'''
def lkdata_setup(file, time_key = 'TIME'):

    hdu = fitsio.FITS(file)

    # TODO: how do I deal with this weird aperture style?
    aperture = hdu[2].read()

    # Collect meta data from all extensions
    meta = {}
    # Get information from the primary header
    for i in range(len(hdu)):
        temp_meta = {}
        head = hdu[i].read_header()
        for k in head.keys():
            temp_meta[k] = head[k]
        meta[f"header{i}"] = temp_meta


    # Get data from the first extension
    time = hdu[1][time_key][:].byteswap().newbyteorder()
    nanmask = np.isnan(time)
    ds = {}
    for tup in spoc_data:
        if tup[1]:
                ds[tup[0]] = DataSeries(data = hdu[1][tup[0]][:].byteswap().newbyteorder()[nanmask],
                                uncertainty = hdu[1][tup[1]][:].byteswap().newbyteorder()[nanmask],
                                time_indices = {'time':time[nanmask]}
                    )
        else:
            ds[tup[0]] = DataSeries(data = hdu[1][tup[0]][:].byteswap().newbyteorder()[nanmask],
                                time_indices = {'time':time[nanmask]}
                    )   

    
    bws = {}
    for k in spoc_bitwise:
         bws[k] = BitwiseSeries(data = hdu[1][k][:].byteswap().newbyteorder()[nanmask],
                    time_indices = {'time':time[nanmask]})


    bos = {}
    for k in spoc_bool:
        bws[k] = BoolSeries(data = hdu[1][k][:].bytesweap().newbyteorder()[nanmask],
            time_indices = {'time':time[nanmask]})     

    print(ds)
    print(bws)
    print(bos)
    dset = DataSet(data_products=ds, 
        #bool_products=bos,
        bitwise_products=bws,
        #header = meta
        time_indices = {'time':time}
        )

    return dset

lkdata_setup(file)
        