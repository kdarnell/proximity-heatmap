import unittest
from proximity_heatmap import ProximityHeatMap
import geopandas as gpd
import numpy as np
import xarray


class TestProximityHeatMap(unittest.TestCase):
    path = "tests/data/test.shp"

    def test_can_load_file(self):
        ph = ProximityHeatMap(self.path)
        ph._convert_to_geo_df()
        self.assertTrue(isinstance(ph.geo_df, gpd.GeoDataFrame))
        self.assertEqual(ph.geo_df.shape, (20, 29))

    def test_cannot_load_file(self):
        ph = ProximityHeatMap("dummy.shp")
        with self.assertRaises(FileNotFoundError):
            ph._convert_to_geo_df()
    
    def test_can_project_into_new_crs(self):
        ph = ProximityHeatMap(self.path)
        ph._convert_to_geo_df()

        ph2 = ProximityHeatMap(self.path)
        ph2._convert_to_geo_df("EPSG:26910")
        self.assertTrue(ph.geo_df.geometry.equals(ph2.geo_df.geometry))

        ph3 = ProximityHeatMap(self.path)
        ph3._convert_to_geo_df("EPSG:4326")
        self.assertFalse(ph.geo_df.geometry.equals(ph3.geo_df.geometry))

    def test_fails_for_false_crs(self):
        ph = ProximityHeatMap(self.path)
        ph._convert_to_geo_df()

        ph2 = ProximityHeatMap(self.path)
        with self.assertRaises(ValueError):
            ph2._convert_to_geo_df("EPSG:NA")

    def test_can_find_rock_types(self):
        ph = ProximityHeatMap(self.path)
        ph._convert_to_geo_df()
        ph._make_tmp_layer("rock_type", ".*sedimentary rocks", "quartz dioritic intrusive rocks")
        self.assertEqual(
            ph.type_map_,
            {
                "mudstone, siltstone, shale fine clastic sedimentary rocks": 1,
                "undivided sedimentary rocks": 1,
                "quartz dioritic intrusive rocks": 2,
            },
        )
        self.assertEqual((ph.geo_df["new_type"] == 2).sum(), 2)
        self.assertEqual((ph.geo_df["new_type"] == 1).sum(), 7)
        self.assertFalse(ph.geo_df["new_type"].isna().any())

    def test_can_generate_raster(self):
        ph = ProximityHeatMap(self.path)
        ph.generate_twoclass_raster(
            layer="rock_type",
            class_one=".*sedimentary rocks",
            class_two="quartz dioritic intrusive rocks",
            resolution=2000,
        )
        self.assertEqual(ph.raster.new_type.shape, (84, 52))
        self.assertTrue(isinstance(ph.raster, xarray.Dataset))

    def test_can_generate_proximity_matrix(self):
        ph = ProximityHeatMap(self.path)
        ph.generate_twoclass_raster(
            layer="rock_type",
            class_one=".*sedimentary rocks",
            class_two="quartz dioritic intrusive rocks",
            resolution=2000,
        )
        _indices = [(ii, jj) for ii in range(ph.raster.new_type.shape[0]) for jj in range(ph.raster.new_type.shape[1])]
        ph.make_twoclass_proximity_array(length=1e5, indices=_indices)
        self.assertEqual(ph.computed_proximity_array_.shape, (84, 52))
        self.assertFalse(np.isnan(ph.computed_proximity_array_).all())

    def test_can_generate_heatmap_probs(self):
        ph = ProximityHeatMap(self.path)
        ph.generate_twoclass_raster(
            layer="rock_type",
            class_one=".*sedimentary rocks",
            class_two="quartz dioritic intrusive rocks",
            resolution=2000,
        )
        _indices = [(ii, jj) for ii in range(ph.raster.new_type.shape[0]) for jj in range(ph.raster.new_type.shape[1])]
        ph.make_twoclass_proximity_array(length=1e5, indices=_indices)
        ph.make_twoclass_heatmap(length=1e5)
        self.assertEqual(ph.heatmap.shape, (84, 52))
        self.assertFalse(np.isnan(ph.heatmap).all())

if __name__ == '__main__':
    unittest.main()