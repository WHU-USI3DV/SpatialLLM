import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

import geopandas as gpd
import pyproj
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from simplification.cutil import simplify_coords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OSMFeatureExtractor:
    """
    Extracts and processes geographic features from OpenStreetMap (OSM) Shapefiles.
    
    This class handles:
    1. Reading Shapefiles from a directory.
    2. Projecting geometries for accurate metric calculations (Area/Length).
    3. Transforming coordinates to WGS84 (Lat/Lon) for output.
    4. Exporting data to a structured JSON format.
    """

    def __init__(self, 
                 shp_dir: str, 
                 output_path: str, 
                 input_crs: int = 4326, 
                 projected_crs: int = 32649):
        """
        Initialize the extractor.

        Args:
            shp_dir (str): Directory containing .shp files.
            output_path (str): Path to save the output JSON.
            input_crs (int): EPSG code for input/output coordinates (default: 4326/WGS84).
            projected_crs (int): EPSG code for metric calculations (default: 32649/UTM Zone 49N).
        """
        self.shp_dir = Path(shp_dir)
        self.output_path = Path(output_path)
        self.input_crs = pyproj.CRS.from_epsg(input_crs)
        self.projected_crs = pyproj.CRS.from_epsg(projected_crs)
        
        # Define layers to process based on OSM naming conventions
        self.target_layers = {'buildings', 'landuse', 'pois', 'roads', 'points', 'water'}
        print(f"Attention: You are process {self.target_layers} layers from OSM data.")

    def _get_geometry_type(self, gdf: gpd.GeoDataFrame) -> str:
        """Determines the geometry type of the GeoDataFrame."""
        # Find the first valid geometry to check type
        first_valid = next((g for g in gdf.geometry if g is not None and not g.is_empty), None)
        
        if not first_valid:
            return "unknown"

        if isinstance(first_valid, (Point, MultiPoint)):
            return "Point"
        elif isinstance(first_valid, (LineString, MultiLineString)):
            return "Polyline"
        elif isinstance(first_valid, (Polygon, MultiPolygon)):
            return "Polygon"
        else:
            raise ValueError(f"Unknown geometry type: {type(first_valid)}")

    def _simplify_coords(self, coords: List[tuple], tolerance: float = 0.0001, precision: int = 5) -> List[List[float]]:
        """
        Simplifies coordinate lists and rounds them.
        
        Args:
            coords: List of (x, y) tuples.
            tolerance: Simplification tolerance (higher = fewer points).
            precision: Decimal places to keep.
        """
        if not coords:
            return []
        simplified = simplify_coords(coords, tolerance)
        return [[round(x, precision), round(y, precision)] for x, y in simplified]

    def load_shapefiles(self) -> Dict[str, List[tuple]]:
        """
        Iterates through the directory and loads relevant shapefiles.
        
        Returns:
            Dict mapping geometry types ('Polygon', 'Polyline', etc.) to lists of (filename, GeoDataFrame).
        """
        if not self.shp_dir.exists():
            raise FileNotFoundError(f"Shapefile directory not found: {self.shp_dir}")

        classified_data = defaultdict(list)
        shp_files = list(self.shp_dir.glob('*.shp'))
        
        logger.info(f"Found {len(shp_files)} shapefiles in {self.shp_dir}")

        for shp_file in shp_files:
            file_name = shp_file.stem
            # Simple filter: check if filename overlaps with target layers
            if not any(layer in file_name.lower() for layer in self.target_layers):
                continue

            try:
                gdf = gpd.read_file(shp_file)
                if gdf.empty:
                    continue
                
                geo_type = self._get_geometry_type(gdf)
                if geo_type != "unknown":
                    classified_data[geo_type].append((file_name, gdf))
                    logger.info(f"Loaded layer: {file_name} ({geo_type})")
            except Exception as e:
                logger.error(f"Failed to read {shp_file}: {e}")

        return dict(classified_data)

    def process_polygons(self, gdf: gpd.GeoDataFrame, use_bbox: bool = True) -> List[Dict]:
        """
        Process polygon layers (e.g., buildings, landuse).
        Calculates area in projected CRS and extracts bbox/coordinates in input CRS.
        """
        features = []
        # Transform to target CRSs once for efficiency
        gdf_wgs = gdf.to_crs(self.input_crs)
        gdf_proj = gdf.to_crs(self.projected_crs)

        for idx, row in gdf.iterrows():
            geom_wgs = gdf_wgs.geometry.iloc[idx]
            if geom_wgs is None or geom_wgs.is_empty:
                continue

            # Calculate area using projected coordinates (meters)
            area = gdf_proj.geometry.iloc[idx].area
            
            osm_id = str(row.get('osm_id', str(idx)))
            feature_data = {
                'name': row.get('name', ''),
                'fclass': f"{row.get('fclass', '')} {row.get('type', '')}".strip(),
                'area': f"{int(area)} m2",
                'center': [round(geom_wgs.centroid.x, 5), round(geom_wgs.centroid.y, 5)]
            }

            if use_bbox:
                minx, miny, maxx, maxy = geom_wgs.bounds
                feature_data['bbox'] = [
                    [round(minx, 5), round(miny, 5)], 
                    [round(maxx, 5), round(maxy, 5)]
                ]
            else:
                # Extract exterior coordinates
                if geom_wgs.geom_type == 'Polygon':
                    coords = list(geom_wgs.exterior.coords)
                    feature_data['nodes'] = self._simplify_coords(coords)
                elif geom_wgs.geom_type == 'MultiPolygon':
                    # For MultiPolygon, take the largest polygon or list all
                    feature_data['nodes'] = [
                        self._simplify_coords(list(p.exterior.coords)) for p in geom_wgs.geoms
                    ]

            features.append({osm_id: feature_data})

        return features

    def process_polylines(self, gdf: gpd.GeoDataFrame) -> List[Dict]:
        """
        Process polyline layers (e.g., roads).
        Calculates length in projected CRS.
        """
        features = []
        gdf_wgs = gdf.to_crs(self.input_crs)
        gdf_proj = gdf.to_crs(self.projected_crs)

        for idx, row in gdf.iterrows():
            geom_wgs = gdf_wgs.geometry.iloc[idx]
            if geom_wgs is None or geom_wgs.is_empty:
                continue

            # Skip unclassified roads without names to reduce noise
            if row.get("fclass") == 'unclassified' and not row.get("name"):
                continue

            length = gdf_proj.geometry.iloc[idx].length
            
            # Extract coordinates
            coords = []
            if geom_wgs.geom_type == 'LineString':
                coords = list(geom_wgs.coords)
            elif geom_wgs.geom_type == 'MultiLineString':
                # Simplified: take the first line or merge them
                coords = list(geom_wgs.geoms[0].coords)

            if not coords:
                continue

            osm_id = str(row.get('osm_id', str(idx)))
            feature_data = {
                'name': row.get('name', ''),
                'fclass': row.get('fclass', ''),
                'length': f"{int(length)} m",
                'nodes': self._simplify_coords(coords)
            }

            features.append({osm_id: feature_data})

        return features

    def process_points(self, gdf: gpd.GeoDataFrame) -> List[Dict]:
        """
        Process point layers (e.g., POIs).
        """
        features = []
        gdf_wgs = gdf.to_crs(self.input_crs)

        for idx, row in gdf.iterrows():
            geom_wgs = gdf_wgs.geometry.iloc[idx]
            if geom_wgs is None or geom_wgs.is_empty:
                continue
            
            if row.get('fclass') in ['crossing', 'traffic_signals']:
                continue

            osm_id = str(row.get('osm_id', str(idx)))
            feature_data = {
                'name': row.get('name', ''),
                'fclass': row.get('fclass', ''),
                'coordinates': [round(geom_wgs.x, 5), round(geom_wgs.y, 5)]
            }

            features.append({osm_id: feature_data})

        return features

    def run(self):
        """Main execution method."""
        osm_data = self.load_shapefiles()
        output_data = {"Polygon": [], "Polyline": [], "Point": []}

        logger.info("Processing features...")
        
        for geo_type, data_list in osm_data.items():
            for _, gdf in data_list:
                if geo_type == "Polygon":
                    # Set use_bbox=False if you want detailed polygon nodes instead of bbox
                    output_data["Polygon"].extend(self.process_polygons(gdf, use_bbox=True))
                elif geo_type == "Polyline":
                    output_data["Polyline"].extend(self.process_polylines(gdf))
                elif geo_type == "Point":
                    output_data["Point"].extend(self.process_points(gdf))

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=1)
        
        logger.info(f"Successfully processed {sum(len(v) for v in output_data.values())} features.")
        logger.info(f"Output saved to: {self.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract geographic features from OSM Shapefiles to JSON.")
    
    parser.add_argument('--shp_path', type=str, required=True, 
                        help="Path to the directory containing .shp files.")
    parser.add_argument('--output', type=str, default="osm_features.json", 
                        help="Path for the output JSON file.")
    parser.add_argument('--input_crs', type=int, default=4326, 
                        help="EPSG code for input/output coordinates (default: 4326/WGS84).")
    parser.add_argument('--projected_crs', type=int, default=32649, 
                        help="EPSG code for projection used in metric calculations (default: UTM Zone 49N).")

    args = parser.parse_args()

    extractor = OSMFeatureExtractor(
        shp_dir=args.shp_path,
        output_path=args.output,
        input_crs=args.input_crs,
        projected_crs=args.projected_crs
    )
    
    extractor.run()