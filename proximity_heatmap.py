import geopandas as gpd
import numpy as np
from geocube.api.core import make_geocube
import re
import typing as t
import xarray
from fiona.errors import DriverError
from pyproj.exceptions import CRSError
from geocube.api.core import make_geocube
from scipy.spatial import distance_matrix, KDTree

INDEX_LIKE = t.Union[t.List[t.Tuple[int]], t.Tuple[t.Tuple[int]], np.ndarray]


class ProximityHeatMap:
    def __init__(self, input_file: str):
        """Class for extracting proximity measurements from a shapefile, geojson, or other file containing geometries

        Parameters:
        input_file: str
            path to shapefile, geojson, etc.
        """

        self.input_file = input_file

    def _convert_to_geo_df(self, crs: t.Optional[str] = None):
        """Convert the shapefile into a geopandas dataframe

        Parameters:
        crs: str
            A valid coordinate reference system for calculations
        """
        try:
            self.geo_df = gpd.read_file(self.input_file)
        except DriverError:
            raise FileNotFoundError(f"No such file or directory called {self.input_file}.")
        if crs is not None:
            try:
                self.geo_df = self.geo_df.to_crs(crs)
            except CRSError:
                raise ValueError(f"{crs} is not a valid crs in pyproj")
        return self

    def _make_tmp_layer(
        self,
        layer_name: t.Union[str, t.Pattern[str]],
        class_one: t.Union[str, t.Pattern[str]],
        class_two: t.Union[str, t.Pattern[str]],
    ):
        """Generate a new layer from a layer using two classes or patterns matching items in that layer

        Parameters:
        layer_name: str
            Name of the layer in the shapefile
        class_one: str or pattern
            Name or pattern to match within layer
        class_two: str or pattern
            Name or pattern to match within layer
        """
        if not hasattr(self, "geo_df"):
            raise Exception("Input file has not been converted into a geopandas dataframe yet.")

        if layer_name not in self.geo_df.columns:
            raise ValueError(f"{layer_name} is not in the input file")
        layer_types = self.geo_df[layer_name].unique()
        class_one_regex = re.compile(class_one)
        class_one_matches = [x for x in layer_types if class_one_regex.match(x)]
        class_two_regex = re.compile(class_two)
        class_two_matches = [x for x in layer_types if class_two_regex.match(x)]
        if class_one_matches is None:
            raise ValueError(f"{class_one} returns zero matches in {layer_name}")
        if class_one_matches is None:
            raise ValueError(f"{class_two} returns zero matches in {layer_name}")
        self.type_map_ = {x: 1 for x in class_one_matches}
        self.type_map_.update({x: 2 for x in class_two_matches})
        self.geo_df = self.geo_df.assign(
            new_type=self.geo_df["rock_type"].apply(lambda x: self._get_type_number(self.type_map_, x))
        )
        return self

    def rasterize_layer(
        self,
        layer: str,
        geometry_layer: str,
        resolution: t.Union[int, float],
        *,
        query: t.Optional[str] = "",
    ) -> xarray.Dataset:
        """Rasterize a layer in the geopandas dataframe

        Parameters:
        layer: str
            Name of the layer to rasterize
        geometry_layer: str
            Name of the geometry layer to use for the rasterization
        resolution: int
            Resolution of the resulting raster
        query: Optional[str]
            Query to pass to geopandas dataframe for filtering data

        Returns:
            Rasterized input layer as xarray.Dataset
        """
        return make_geocube(
            vector_data=self.geo_df.query(f"{query}")[[layer, geometry_layer]],
            resolution=(-resolution, resolution),
        )

    def generate_twoclass_raster(
        self,
        layer: str,
        class_one: t.Union[str, t.Pattern[str]],
        class_two: t.Union[str, t.Pattern[str]],
        resolution: t.Union[int, float],
    ):
        """Generate raster from a layer by matching classes in layer and converting matches to 1 or 2

        Paramters:
        layer: str
            Name of layer
        class_one: str or pattern
            Matching items in layer
        class_two: str or pattern
            Matching items in layer
        resolution: int
            Resolution
        """
        self.layer_ = layer
        self.class_one_ = class_one
        self.class_two_ = class_two
        self.resolution_ = resolution
        self._convert_to_geo_df()
        self._make_tmp_layer(layer, class_one, class_two)
        self.raster = self.rasterize_layer("new_type", "geometry", resolution, query="new_type != 0")
        return self

    def make_twoclass_proximity_array(
        self,
        *,
        length: t.Union[int, float] = 1e4,
        indices: t.Optional[INDEX_LIKE] = None,
    ):
        """Generate two class proximity array after pre-computing raster

        Parameters:
        length: Union[int, float]
            length scale for generating small computation matrix
        indices: INDEX_LIKE
            set of indices in self.raster where we are interested in generating heatmap
        """
        if hasattr(self, "raster"):
            labels = self.raster.new_type.values
            x_coords = self.raster.x.values
            y_coords = self.raster.y.values
            self.computed_proximity_array_ = self.proximity_array(
                x_coords, y_coords, labels, length, self.resolution_, indices=indices
            )
            return self
        else:
            raise Exception("No rasterized layer. Run `.generate_twoclass_raster` first")

    def make_twoclass_heatmap(self, *, length: t.Union[int, float] = 1e4):
        """Generate two class heatmap after pre-computing raster and proximity matrix

        Parameters:
        length: Union[int, float]
            length scale for generating small computation matrix
        """
        if hasattr(self, "computed_proximity_array_"):
            self.heatmap = self.make_heatmap(
                np.sqrt(self.computed_proximity_array_),
                gaussian,
                scale=(1 / 0.25 / length) ** 2,
            )
            out_grid = self.raster.copy()
            out_grid["heat_map"] = self.raster.new_type.copy()
            out_grid["heat_map"].values = np.where(self.heatmap > 0, self.heatmap, np.nan)
            self.heatmap_xarray = out_grid["heat_map"]
        else:
            raise Exception("No computed_proximity_array_. Run `.make_twoclass_proximity_array` first")
        return self

    @staticmethod
    def make_heatmap(proximity_array, probability_function: t.Callable, **kwargs):
        """Generic function for transforming a proximity_array by the callable `probability_function`

        Parameters:
        proximity_array:
            data in
        probabiliy_function: Callable
            transformation function
        """
        return probability_function(proximity_array, **kwargs)

    @staticmethod
    def proximity_array(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        labels: np.ndarray,
        length: t.Union[int, float],
        resolution: t.Union[int, float],
        *,
        indices: t.Optional[INDEX_LIKE] = None,
    ) -> np.ndarray:
        """Generic function for generating proximity array from a labeled array and coordinates

        Parameters:
        x_coords: np.ndarray
            coordinates in x dimension (assumed to be 1-d)
        y_coords: np.ndarray
            coordinates in y dimension (assumed to be 1-d)
        labels: np.ndarray
            labeled data where labels of interest are 1 and 2 and all other data is np.nan (assumed to be 2-d)
        length: Union[int, float]
            length scale for generating small computation matrix
        resolution: Union[int, float]
            resolution of regular grid (assumed to be identical for x and y)
        indices: Optional[INDEX_LIKE]
            set of indices in labels where we are interested in generating heatmap

        Returns:
        total_dist: np.ndarray
            proximity matrix of the two classes with shape equivalent to labels
        """
        buffer = int((length / 2) // resolution)
        total_dist = labels * np.nan
        if indices is None:
            x_min, x_max, y_min, y_max = _get_trimmed_extent(labels, buffer)
            indices = [(ii, jj) for ii in range(y_min, y_max) for jj in range(x_min, x_max)]
        if isinstance(indices, np.ndarray):
            if indices.ndim == 1:
                raise ValueError("Indices must be an iterable of tuples or a 2-d np.ndarray")
        x_mat = x_coords[np.newaxis, :] * np.ones_like(labels)
        y_mat = y_coords[:, np.newaxis] * np.ones_like(labels)
        for ii, jj in indices:
            if np.isnan(total_dist[ii, jj]):
                _max_x = np.minimum(total_dist.shape[1], jj + buffer)
                _max_y = np.minimum(total_dist.shape[0], ii + buffer)
                _min_x = np.maximum(0, jj - buffer)
                _min_y = np.maximum(0, ii - buffer)
                tmp_array = labels[_min_y:_max_y, _min_x:_max_x]
                if (tmp_array == 1).any() and (tmp_array == 2).any():
                    _x = x_mat[_min_y:_max_y, _min_x:_max_x]
                    _y = y_mat[_min_y:_max_y, _min_x:_max_x]
                    min_dist_1 = np.min(
                        np.sqrt(
                            (x_coords[jj] - _x[tmp_array == 1]) ** 2
                            + (y_coords[ii] - _y[tmp_array == 1]) ** 2
                        )
                    )
                    min_dist_2 = np.min(
                        np.sqrt(
                            (x_coords[jj] - _x[tmp_array == 2]) ** 2
                            + (y_coords[ii] - _y[tmp_array == 2]) ** 2
                        )
                    )
                    total_dist[ii, jj] = min_dist_1 ** 2 + min_dist_2 ** 2
        return total_dist

    @staticmethod
    def _get_type_number(type_map, x):
        """Helper utilty for encoding a new layer in shapefile"""
        if x in type_map.keys():
            return type_map[x]
        else:
            return 0

class KDTreeProximityHeatMap(ProximityHeatMap):
    @staticmethod
    def proximity_array(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        labels: np.ndarray,
        length: t.Union[int, float],
        resolution: t.Union[int, float],
        indices: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generic function for generating proximity array from a labeled array and 
        coordinates that is optimized by scipy.spatial.KDTree 

        Parameters:
        x_coords: np.ndarray
            coordinates in x dimension (assumed to be 1-d)
        y_coords: np.ndarray
            coordinates in y dimension (assumed to be 1-d)
        labels: np.ndarray
            labeled data where labels of interest are 1 and 2 and all other data is np.nan (assumed to be 2-d)
        length: Union[int, float]
            length scale for generating kernel
        resolution: Union[int, float]
            resolution of regular grid (assumed to be identical for x and y)
        indices: np.ndarray of bools
            where True, compute heatmap
            
        Returns:
        total_dist: np.ndarray
            proximity matrix of the two classes as shape equivalent to labels
        """
        x_mat = x_coords[np.newaxis, :] * np.ones_like(labels)
        y_mat = y_coords[:, np.newaxis] * np.ones_like(labels)
        coords = np.vstack((x_mat.ravel(), y_mat.ravel())).T
        full_labels = labels.ravel()
        if indices is None:
            indices = np.ones_like(labels, dtype=bool)
        elif not isinstance(indices, np.ndarray) or indices.dtype != bool:
            raise ValueError("indices should be a boolean array equal in shape to labels")
        kd_tree_one = KDTree(coords[full_labels == 1, :])
        kd_tree_two = KDTree(coords[full_labels == 2, :])
        query_pts = np.vstack((x_mat[indices].ravel(), y_mat[indices].ravel())).T
        dist_1 = np.zeros(len(query_pts)) * np.nan
        dist_2 = np.zeros(len(query_pts)) * np.nan
        _increment = 1000 # This is configurable.
        for ii in range(0, len(query_pts), _increment):
            _end = np.minimum(len(query_pts), ii + _increment)
            dist_1[ii: _end] = kd_tree_one.query(query_pts[ii: _end], k=1, distance_upper_bound=length)[0]
            dist_2[ii: _end] = kd_tree_two.query(query_pts[ii: _end], k=1, distance_upper_bound=length)[0]
        _total_dist = dist_1**2 + dist_2**2
        total_dist = np.zeros_like(labels) * np.nan
        total_dist[indices] = _total_dist
        return total_dist




def gaussian(data, scale):
    """Gaussian transformation

    Parameters:
    data
        data to transform
    scale
        scalar value
    """
    return np.exp(-scale * data ** 2)


def _get_trimmed_extent(labels: np.ndarray, buffer: int):
    """Helper function for trimming labels array to extent where they both exist

    Parameters:
    labels: np.ndarray
        matrix of labels where relevant labels are 1 or 2
    buffer: int
        width in indices to search

    Returns:
    bounding indices where x* is assumed be measured along axis 1 and y* is assumed to be measured along axis 0
    """
    x_extents = np.argwhere((labels == 1).any(axis=0) & (labels == 2).any(axis=0))[[0, -1]]
    y_extents = np.argwhere((labels == 1).any(axis=1) & (labels == 2).any(axis=1))[[0, -1]]
    x_min = int(np.maximum(0, x_extents[0] - buffer))
    x_max = int(np.minimum(labels.shape[1], x_extents[1] + buffer))
    y_min = int(np.maximum(0, y_extents[0] - buffer))
    y_max = int(np.minimum(labels.shape[0], y_extents[1] + buffer))
    return x_min, x_max, y_min, y_max