"""
Class and functions for reading S2 images.

https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library

"""
from rasterio import windows, warp, features, coords
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, MultiPolygon
from lxml.etree import parse
import xml.etree.ElementTree as ET
import rasterio
import h5py

import numpy as np
import dl_l8s2_uv.utils as misc

import os
import datetime
import logging

BANDS_RGB = [3, 2, 1]
BANDS_10M = [1, 2, 3, 7]
BANDS_20M = [4, 5, 6, 8, 11, 12]
BANDS_60M = [0, 9, 10]
BANDS_RES = {"10": BANDS_10M, "20": BANDS_20M, "60": BANDS_60M}
BANDS_LIST = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
              "B08": 7, "B8A": 8, "B09": 9, "B10": 10, "B11": 11, "B12": 12}
BAND_PATTERN = "\D{1}\d{2}\D{3}_\d{8}\D{1}\d{6}_B(\w{2})"


class S2Image:
    def __init__(self, s2_folder, slice_rows_cols=None, size_def="30"):
        self.folder = s2_folder
        self.name = os.path.splitext(os.path.basename(self.folder))[0]
        self.satname = "S2"
        self.content = np.array(misc.list_dir(self.folder, []))
        self.band_names, self.band_paths = self.list_bands()
        self.metadata_msi = misc.search_string_list(self.content, "MTD_MSIL1C")
        self.metadata_tl = misc.search_string_list(self.content, "MTD_TL")
        self.read_metadata()
        self.crs = {'init': 'epsg:' + str(self.epsg)}
        self.transform_dict = self.load_transform(self.band_paths)
        self.transform_dict.update({"30": [rasterio.Affine(30, self.transform_dict["60"][0].b, self.transform_dict["60"][0].c,
                                                           self.transform_dict["60"][0].d, -30, self.transform_dict["60"][0].f)]})
        self.dimsByRes.update({"30": tuple([2 * x for x in self.dimsByRes["60"]])})
        self.granule = self.sorted_bands(self.band_paths)
        self.out_res = size_def
        self.rasterio_transform = self.transform_dict[self.out_res][0]

        if slice_rows_cols is None:
            slice_ = (slice(None), slice(None))
        else:
            slice_ = tuple(slice_rows_cols)

        self.slice = slice_

        logging.basicConfig(level=logging.WARNING)

    def set_size_default(self, size_def):
        self.out_res = size_def
        self.set_slice(None)

    @property
    def nrows(self):
        return self.dimsByRes[self.out_res][0]

    @property
    def ncols(self):
        return self.dimsByRes[self.out_res][1]

    @property
    def polygon(self):
        with rasterio.open(self.granule[0], driver='JP2OpenJPEG') as src:
            bbox = src.bounds

        bbox_lnglat = warp.transform_bounds(self.crs,
                                            {'init': 'epsg:4326'},
                                            *bbox)
        return generate_polygon(bbox_lnglat)

    @property
    def transform(self):
        tr = self.transform_dict[self.out_res][0]
        return np.array([[tr.a, tr.b, tr.c], [tr.d, tr.e, tr.f]])

    def __str__(self):
        return self.folder

    def set_slice(self, slice_):
        if slice_ is None:
            slice_ =  (slice(None), slice(None))

        self.slice = slice_

    def load_bands(self, bands=None, slice_=None, enabled_cache=True):
        """
        Expected slice_ to match self.out_res resolution

        :param bands:
        :param slice_:
        :param enabled_cache:
        :return:
        """
        if bands is None:
            bands = list(range(len(BANDS_LIST)))
        assert any([type(bands) == np.ndarray, type(bands) == list]), "Selected bands must be an array or list"

        assert (self.out_res in ["10", "20", "30", "60"]), \
            "Not valid output resolution. \n" "Choose ""10"", ""20"", ""30"", ""60"""

        bbox, slice_ = self.from_slice_out_res(slice_=slice_)

        list_res = np.ndarray((len(bands),), dtype=np.int32)
        for i, b in enumerate(bands):
            list_res[i] = [outres for outres, list_bands in BANDS_RES.items() if b in list_bands][0]

        img = None
        for id_, b in enumerate(bands):
            if self.check_cache(band_name=misc.getKeyByValue(BANDS_LIST, b)) and enabled_cache:
                img_read, mask_read = self.read_band(band_name=misc.getKeyByValue(BANDS_LIST, b), slice_=slice_)
            else:
                with rasterio.open(self.granule[b], driver='JP2OpenJPEG') as src:
                    window_read_slices = windows.from_bounds(*bbox, src.transform).toslices()
                    window_read = windows.Window.from_slices(*misc.round_slice(window_read_slices, mode="outer"))
                    band = src.read(1, window=window_read)
                    mask = np.bitwise_or(band == 0, band == (2**16)-1)

                    if b not in BANDS_RES.get(self.out_res, []):
                        if np.any([np.int(bbox[i + 2] - bbox[i]) % np.max(list_res) != 0 for i in range(0, 2)]) and id_ == 0:
                            logging.warning("The corresponding slice at different resolution does not match precisely."
                                            " Better choose a divisible slice by {} "
                                            "in order to get exact resampled bands".format(np.max(list_res)))

                        transform = windows.transform(window_read, src.transform)

                        # Reproject to self.out_res resolution
                        img_read = self.reproject_band(band, transform, bbox)
                        mask_read = self.reproject_band(mask.astype(np.float32), transform, bbox)
                        mask_read = mask_read > 1e-6
                    else:
                        img_read = band
                        mask_read = mask

            if img is None:
                img = np.ndarray(shape=img_read.shape+(len(bands),), dtype=img_read.dtype)
                mask_img = np.ndarray(shape=mask_read.shape+(len(bands),), dtype=mask_read.dtype)

            img[..., id_] = img_read
            mask_img[..., id_] = mask_read

        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=2)
            mask_img = np.squeeze(mask_img, axis=2)

        if img.dtype != np.float32:
            img = (img / 1e4).astype(np.float32)

        img = np.ma.MaskedArray(img, mask=mask_img)

        return img

    def crs_proj(self):
        with rasterio.open(self.granule[1], "r") as src:
            src_crs = src.crs
        return src_crs

    def src_rasterio(self):
        return rasterio.open(self.granule[1], "r")

    def load_mask(self, slice_=None):
        if self.out_res != '30':
            b_ind = BANDS_RES[self.out_res][0]
        else:
            b_ind = 0

        band = self.load_bands(bands=[b_ind], slice_=slice_)
        return np.ma.getmaskarray(band)

    def from_slice_out_res(self, slice_=None):
        assert self.out_res in ["10", "30", "20", "60"], "output resolution must be 10, 20, 30 or 60"

        if slice_ is None:
            slice_ = self.slice
        if slice_ is None:
            slice_ = (slice(None), slice(None))

        shape = (self.dimsByRes[self.out_res][0], self.dimsByRes[self.out_res][1])
        slice_norm = []
        for i, s in enumerate(slice_):
            start = 0 if s.start is None else s.start
            end = shape[i] if s.stop is None else s.stop
            slice_norm.append(slice(start, end))
        slice_ = tuple(slice_norm)

        # Assert slice fits the dims
        misc.assert_valid_slice(slice_,
                                         (self.dimsByRes[self.out_res][0], self.dimsByRes[self.out_res][1]))

        bbox = windows.bounds(windows.Window.from_slices(*slice_), self.transform_dict[self.out_res][0])

        return bbox, slice_

    def reproject_band(self, band, src_transform, bbox):
        # shift dst_transform
        # bbox --> (left, bottom, right, top)
        assert self.out_res in ["10", "30", "20", "60"], "output resolution must be 10, 20, 30 or 60"

        transform_out_res = self.transform_dict[self.out_res][0]

        dst_transform = rasterio.Affine(transform_out_res.a,
                                        transform_out_res.b,
                                        bbox[0] if transform_out_res.a > 0 else bbox[2],
                                        transform_out_res.d,
                                        transform_out_res.e,
                                        bbox[3] if transform_out_res.e < 0 else bbox[1])

        window_read = windows.from_bounds(*bbox, dst_transform)
        shape_new = tuple([int(round(s)) for s in windows.shape(window_read)])
        data_new_proj = np.ndarray(shape=shape_new, dtype=band.dtype)

        reproject(
            band,
            data_new_proj,
            src_transform=src_transform,
            src_crs=self.crs,
            dst_transform=dst_transform,
            dst_crs=self.crs,
            resampling=Resampling.cubic_spline)

        return data_new_proj

    def check_cache(self, band_name):
        assert self.out_res in ["10", "20", "30", "60"], "Not valid output resolution. \n" \
                                                    "Choose ""10"", ""20"", ""30"" or ""60"""
        cache_file = misc.search_string_list(self.content, 'IMG_DATA_' + self.out_res + os.sep)
        if cache_file is not None:
            with h5py.File(cache_file, 'r') as input_f:
                keys_ = list(input_f.keys())

            return band_name in keys_

        return False

    def generate_cache_bands(self, overwrite=False):
        """
        Function to generate and store all bands to a fixed resolution in order to speed up reading of bands
        from different resolutions. Resolution is set by <size_def> at init
        :param overwrite: if True, cached bands at Xm resolution will be re-generated
        :return:
        """
        old_slice = self.slice
        self.set_slice((slice(None), slice(None)))
        for b in list(BANDS_LIST.keys()):
            # Gnerate chache in all cases except only exists cache but overwite is False
            if overwrite or not(self.check_cache(band_name=b) or overwrite):
                if overwrite:
                    logging.warning('Overwriting cache: {} ({}m)'.format(b, self.out_res))

                data = self.load_bands(bands=[BANDS_LIST[b]], enabled_cache=False)
                self.save_band(data=data, band_name=b)
                # print('Saved {} - shape: {}'.format(b, data.shape))
                logging.info('Saved cache: {} ({}m) - shape: {}'.format(b, self.out_res, data.shape))
            else:
                logging.warning('Skipping cache: {} ({}m)'.format(b, self.out_res))

        self.content = np.array(misc.list_dir(self.folder, []))
        self.slice = old_slice

    def read_band(self, band_name, slice_=None):
        cache_file = misc.search_string_list(self.content, 'IMG_DATA_' + self.out_res + os.sep)
        assert cache_file is not None, 'Cache file does not exist at requested resolution'
        with h5py.File(cache_file, 'r') as input_f:
            if slice_ is not None:
                band = input_f[band_name][slice_]
            else:
                band = input_f[band_name][:]

        mask = band == -1
        return band, mask

    def save_band(self, data, band_name):
        """
        Save a reprojected band in a HDF5 file
        :param data:
        :param band_name:
        :return:
        """
        io_modes = {0:"w", 1:"r+"}
        assert self.out_res in ["10", "20", "30", "60"], \
            "Not valid output resolution. \n" "Choose ""10"", ""20"", ""30"", ""60"""
        if data.ndim > 2:
            msk = np.any(np.ma.getmaskarray(data), axis=-1, keepdims=False)
        else:
            msk = np.ma.getmaskarray(data)

        data[msk, ...] = -1

        path_folder = os.path.join(os.path.dirname(self.metadata_tl), 'IMG_DATA_'+self.out_res+os.sep)
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        cache_file = os.path.join(path_folder, self.name + '.HDF5')
        with h5py.File(cache_file, io_modes[os.path.exists(cache_file)]) as output:
            shape_expected = self.dimsByRes[self.out_res]
            assert shape_expected == data.shape, "Band shape not expected.\n" \
                                                 "Expected: {} - Reprojected: {}".format(shape_expected, data.shape)
            if band_name in output:
                output[band_name][...] = data
            else:
                output.create_dataset(band_name,
                                      data=data, chunks=(512, 512),
                                      compression="gzip")

    def read_metadata(self):
        '''
        Read metadata TILE to parse information about the acquisition and properties of GRANULE bands
        Source: fmask.sen2meta.py
        :return: relevant attributes
        '''
        with open(self.metadata_tl) as f:
            root = ET.fromstring(f.read())
            # XML namespace prefix
            nsPrefix = root.tag[:root.tag.index('}') + 1]
            nsDict = {'n1': nsPrefix[1:-1]}

            generalInfoNode = root.find('n1:General_Info', nsDict)

            sensingTimeNode = generalInfoNode.find('SENSING_TIME')
            sensingTimeStr = sensingTimeNode.text.strip()
            self.datetime = datetime.datetime.strptime(sensingTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
            tileIdNode = generalInfoNode.find('TILE_ID')
            tileIdFullStr = tileIdNode.text.strip()
            self.tileId = tileIdFullStr.split('_')[-2]
            self.satId = tileIdFullStr[:3]
            self.procLevel = tileIdFullStr[13:16]

            geomInfoNode = root.find('n1:Geometric_Info', nsDict)
            geocodingNode = geomInfoNode.find('Tile_Geocoding')
            epsgNode = geocodingNode.find('HORIZONTAL_CS_CODE')
            self.epsg = epsgNode.text.split(':')[1]

            # Dimensions of images at different resolutions.
            self.dimsByRes = {}
            sizeNodeList = geocodingNode.findall('Size')
            for sizeNode in sizeNodeList:
                res = sizeNode.attrib['resolution']
                nrows = int(sizeNode.find('NROWS').text)
                ncols = int(sizeNode.find('NCOLS').text)
                self.dimsByRes[res] = (nrows, ncols)

            # Upper-left corners of images at different resolutions.
            self.ulxyByRes = {}
            posNodeList = geocodingNode.findall('Geoposition')
            for posNode in posNodeList:
                res = posNode.attrib['resolution']
                ulx = float(posNode.find('ULX').text)
                uly = float(posNode.find('ULY').text)
                self.ulxyByRes[res] = (ulx, uly)

            # Sun and satellite angles.
            tileAnglesNode = geomInfoNode.find('Tile_Angles')
            sunZenithNode = tileAnglesNode.find('Sun_Angles_Grid').find('Zenith')
            self.angleGridXres = float(sunZenithNode.find('COL_STEP').text)
            self.angleGridYres = float(sunZenithNode.find('ROW_STEP').text)
            self.sunZenithGrid = self.makeValueArray(sunZenithNode.find('Values_List'))
            sunAzimuthNode = tileAnglesNode.find('Sun_Angles_Grid').find('Azimuth')
            self.sunAzimuthGrid = self.makeValueArray(sunAzimuthNode.find('Values_List'))
            self.anglesGridShape = self.sunAzimuthGrid.shape

            # Viewing angle per grid cell, from the separate layers
            # given for each detector for each band.
            viewingAngleNodeList = tileAnglesNode.findall('Viewing_Incidence_Angles_Grids')
            self.viewZenithDict = self.buildViewAngleArr(viewingAngleNodeList, 'Zenith')
            self.viewAzimuthDict = self.buildViewAngleArr(viewingAngleNodeList, 'Azimuth')

            # Coordinates of the angle grids.
            (ulx, uly) = self.ulxyByRes["10"]
            self.anglesULXY = (ulx - self.angleGridXres / 2.0, uly + self.angleGridYres / 2.0)

    def buildViewAngleArr(self, viewingAngleNodeList, angleName):
        """
        Viewing angle array from the detector strips given as
        separate arrays.

        :param viewingAngleNodeList: incidence angle array from metadata
        :param angleName: 'Zenith' or 'Azimuth'.
        :return: dictionary of 2-d arrays, keyed by the bandId string.
        """
        angleArrDict = {}
        for viewingAngleNode in viewingAngleNodeList:
            bandId = viewingAngleNode.attrib['bandId']
            angleNode = viewingAngleNode.find(angleName)
            angleArr = self.makeValueArray(angleNode.find('Values_List'))
            if bandId not in angleArrDict:
                angleArrDict[bandId] = angleArr
            else:
                mask = (~np.isnan(angleArr))
                angleArrDict[bandId][mask] = angleArr[mask]
        return angleArrDict

    def get_polygons_bqa(self):
        def polygon_from_coords(coords, fix_geom=False, swap=True, dims=2):
            """
            Return Shapely Polygon from coordinates.
            - coords: list of alterating latitude / longitude coordinates
            - fix_geom: automatically fix geometry
            """
            assert len(coords) % dims == 0
            number_of_points = int(len(coords) / dims)
            coords_as_array = np.array(coords)
            reshaped = coords_as_array.reshape(number_of_points, dims)
            points = [(float(i[1]), float(i[0])) if swap else ((float(i[0]), float(i[1]))) for i in reshaped.tolist()]
            polygon = Polygon(points).buffer(0)
            try:
                assert polygon.is_valid
                return polygon
            except AssertionError:
                if fix_geom:
                    return polygon.buffer(0)
                else:
                    raise RuntimeError("Geometry is not valid.")

        exterior_str = str("eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList")
        interior_str = str("eop:extentOf/gml:Polygon/gml:interior/gml:LinearRing/gml:posList")
        gml = misc.search_string_list(self.content, 'MSK_CLOUDS_B00.gml')
        root = parse(gml).getroot()
        nsmap = {k: v for k, v in root.nsmap.items() if k}
        try:
            for mask_member in root.iterfind("eop:maskMembers", namespaces=nsmap):
                for feature in mask_member:
                    type = feature.findtext("eop:maskType", namespaces=nsmap)

                    ext_elem = feature.find(exterior_str, nsmap)
                    dims = int(ext_elem.attrib.get('srsDimension', '2'))
                    ext_pts = ext_elem.text.split()
                    exterior = polygon_from_coords(ext_pts, fix_geom=True, swap=False, dims=dims)
                    try:
                        interiors = [polygon_from_coords(int_pts.text.split(), fix_geom=True, swap=False, dims=dims)
                                     for int_pts in feature.findall(interior_str, nsmap)]
                    except AttributeError:
                        interiors = []

                    yield dict(geometry=Polygon(exterior, interiors).buffer(0),
                               attributes=dict(maskType=type),
                               interiors=interiors)

        except StopIteration:
            yield dict(geometry=Polygon(),
                       attributes=dict(maskType=None),
                       interiors=[])
            raise StopIteration()

    def load_clouds_bqa(self, slice_=None):
        """
        Load BQA mask stored as polygons in metadata.
        :param slice_:
        :return: L1C cloud mask
        """
        mask_types = ["OPAQUE", "CIRRUS"]
        poly_list = list(self.get_polygons_bqa())

        nrows = self.nrows
        ncols = self.ncols
        transform_ = self.transform_dict[self.out_res][0]
        _, slice_ = self.from_slice_out_res(slice_=slice_)

        def get_mask(mask_type=mask_types[0]):
            assert mask_type in mask_types, "mask type must be OPAQUE or CIRRUS"
            fill_value = {m: i+1 for i, m in enumerate(mask_types)}
            n_polys = np.sum([poly["attributes"]["maskType"] == mask_type for poly in poly_list])
            msk = np.zeros(shape=(nrows, ncols), dtype=np.float32)
            if n_polys > 0:
                multi_polygon = MultiPolygon([poly["geometry"]
                                              for poly in poly_list
                                              if poly["attributes"]["maskType"] == mask_type]).buffer(0)
                bounds = multi_polygon.bounds
                bbox2read = coords.BoundingBox(*bounds)
                window_read = windows.from_bounds(*bbox2read, transform_)
                slice_read = tuple(slice(int(round(s.start)), int(round(s.stop))) for s in window_read.toslices())
                out_shape = tuple([s.stop - s.start for s in slice_read])
                transform_slice = windows.transform(window_read, transform_)

                shapes = [({"type": "Polygon",
                            "coordinates": [np.stack([
                                p_elem["geometry"].exterior.xy[0],
                                p_elem["geometry"].exterior.xy[1]], axis=1).tolist()]}, fill_value[mask_type])
                          for p_elem in poly_list if p_elem["attributes"]['maskType'] == mask_type]
                sub_msk = features.rasterize(shapes=shapes, fill=0,
                                             out_shape=out_shape, dtype=np.float32,
                                             transform=transform_slice)
                msk[slice_read] = sub_msk

            return msk

        mask = self.load_mask(slice_=slice_)
        if getattr(self, 'load_clouds', None) is not None:
            gt = self.load_clouds(slice_=slice_)
            if gt is not None:
                mask = mask | np.ma.getmask(gt)

        assert mask.shape == misc.shape_slice(slice_), "Different shapes {} {}".format(mask.shape,
                                                                                                misc.shape_slice(
                                                                                                    slice_))
        msk_op_cirr = [np.ma.MaskedArray(get_mask(mask_type=m)[slice_], mask=mask) for m in mask_types]
        msk_clouds = np.ma.MaskedArray(np.clip(np.sum(msk_op_cirr, axis=0), 0, 1), mask=mask)
        return msk_clouds

    def load_rgb(self, slice_=None, with_mask=True):
        rgb = self.load_bands(bands=BANDS_RGB,slice_=slice_)
        if with_mask:
            rgb = rgba(rgb)

        return rgb

    def list_bands(self):
        band_names = dict()
        band_paths = dict()
        for av_res, band_list in BANDS_RES.items():
            bands = [None] * len(band_list)
            band_name = [None] * len(band_list)
            for id_, b in enumerate(band_list):
                band_name[id_] = misc.getKeyByValue(BANDS_LIST, b)
                bands[id_] = misc.search_string_list(self.content, "_"+band_name[id_]+".jp2")

            band_paths.update({av_res: bands})
            band_names.update({av_res: band_name})

        return band_names, band_paths

    def load_transform(self, bands):
        transform = {}
        for av_res, b_res in bands.items():
            transform_list = [None] * len(b_res)
            for id_, b in enumerate(b_res):
                with rasterio.open(b, driver='JP2OpenJPEG') as src:
                    transform_list[id_] = src.transform

            transform.update({av_res: transform_list})
        return transform

    @staticmethod
    def makeValueArray(valuesListNode):
        """
        Take a <Values_List> node from the XML and return an array of the values contained
        within it.
        :return: 2-d numpy array
        """
        valuesList = valuesListNode.findall('VALUES')
        vals = []
        for valNode in valuesList:
            text = valNode.text
            vals.append([np.float32(x) for x in text.strip().split()])

        return np.array(vals)

    @staticmethod
    def affine_transform_asarray(affine_obj):
        """
        Parse an affine object into a numpy array.
        :param affine_obj:
        :return:
        """
        np_transform = np.array([[affine_obj.a, affine_obj.b, affine_obj.c],
                                 [affine_obj.d, affine_obj.e, affine_obj.f]])
        return np_transform

    @staticmethod
    def sorted_bands(band_paths):
        def unpack_list(nested_list):
            unpacked_list = np.array([y for x in nested_list for y in x])
            return unpacked_list

        paths = unpack_list(list(band_paths.values()))
        indices = unpack_list(list(BANDS_RES.values()))
        sorted_bands = [paths[np.argwhere(indices == id_)[0][0]] for id_ in range(len(indices))]
        return sorted_bands


def rgba(bands):
    msk = (~np.any(np.ma.getmaskarray(bands), axis=-1, keepdims=True)).astype(np.float32)
    rgba_ = np.concatenate((bands, msk), axis=-1)
    return rgba_


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]]