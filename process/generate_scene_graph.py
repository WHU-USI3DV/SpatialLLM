import json
import argparse
import logging
import numpy as np
import geopandas as gpd
import pyproj
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPoint, Polygon
from geopy.distance import geodesic
from pyntcloud import PyntCloud
from typing import Dict, List, Tuple, Any

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SceneGraphGenerator:
    def __init__(self, args):
        self.pc_path = Path(args.pc_file)
        self.shp_dir = Path(args.shp_dir)
        # The final output file path (e.g., scene_structured.json)
        self.output_json = Path(args.output_json)
        self.osm_map_file = Path(args.osm_map_file)
        
        # This is the base structural JSON (Input) that needs to be updated
        self.osm_template_file = Path(args.osm_json_file) if args.osm_json_file else None

        # CRS Definitions
        self.input_crs = pyproj.CRS.from_epsg(args.input_crs)       # WGS84
        self.projected_crs = pyproj.CRS.from_epsg(args.projected_crs) # Metric CRS (e.g., UTM)
        
        self.osm_instance_map = {}
        if self.osm_map_file.exists():
            with open(self.osm_map_file, 'r', encoding='utf-8') as f:
                self.osm_instance_map = json.load(f)
        else:
            logger.warning(f"OSM instance map not found at {self.osm_map_file}")

        if not self.osm_template_file or not self.osm_template_file.exists():
            raise FileNotFoundError(f"The input OSM JSON template ({self.osm_template_file}) is required for this pipeline.")

    def load_shapefiles(self, layers: List[str]) -> Dict[str, gpd.GeoDataFrame]:
        """Loads shapefiles and projects them to the metric CRS."""
        data = {}
        for layer in layers:
            shp_path = self.shp_dir / f"{layer}.shp"
            if not shp_path.exists():
                logger.warning(f"Layer not found: {layer}")
                continue
            
            logger.info(f"Loading layer: {layer}")
            gdf = gpd.read_file(shp_path)
            
            # Reproject to metric CRS for accurate distance/area calc
            if gdf.crs:
                gdf = gdf.to_crs(self.projected_crs)
            else:
                gdf.set_crs(self.input_crs).to_crs(self.projected_crs, inplace=True)
                
            data[layer] = gdf
        return data

    def points_to_polygon(self, points: np.ndarray) -> Polygon:
        """Converts WGS84 points to a convex hull in the projected CRS."""
        geometry = [Point(xy) for xy in points]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=self.input_crs)
        gdf_proj = gdf.to_crs(self.projected_crs)
        return MultiPoint(gdf_proj.geometry.tolist()).convex_hull

    def get_height(self, z_values: np.ndarray) -> float:
        """Calculates relative height, excluding outliers."""
        if len(z_values) == 0: return 0.0
        return round(float(np.percentile(z_values, 95) - np.percentile(z_values, 5)), 1)

    def query_nearby_features(self, target_poly: Polygon, 
                            search_layers: Dict[str, gpd.GeoDataFrame], 
                            radius: float) -> str:
        """Finds environmental features (roads, POIs) within a radius."""
        descriptions = []
        target_buffer = target_poly.buffer(radius)
        
        for layer_name, gdf in search_layers.items():
            # Spatial index query for performance
            possible_matches_idx = gdf.sindex.query(target_buffer)
            possible_matches = gdf.iloc[possible_matches_idx]
            
            items = []
            for _, row in possible_matches.iterrows():
                geom = row.geometry
                # Skip self-intersections (mainly for buildings layer against itself)
                if layer_name == 'buildings' and geom.distance(target_poly) < 0.1:
                    continue

                if target_buffer.intersects(geom):
                    dist = target_poly.distance(geom)
                    name = row.get('name', '')
                    fclass = row.get('fclass', layer_name)
                    
                    if name or fclass:
                        desc = f"{name or fclass}"
                        if dist > 1:
                            desc += f" ({int(dist)}m)"
                        items.append(desc)
            
            if items:
                # Limit items to avoid excessive prompt length
                unique_items = list(set(items))[:3] 
                descriptions.append(f"{layer_name.upper()}: {', '.join(unique_items)}")
                
        return " | ".join(descriptions)

    def format_spatial_relation(self, source_center_ll: List[float], target_center_ll: List[float], 
                              target_info: Dict) -> str:
        """Calculates bearing and geodesic distance."""
        # Bearing calculation
        dx = target_center_ll[0] - source_center_ll[0]
        dy = target_center_ll[1] - source_center_ll[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        
        directions = ['East', 'Northeast', 'North', 'Northwest', 
                      'West', 'Southwest', 'South', 'Southeast']
        idx = int((angle + 22.5) / 45) % 8
        direction = directions[idx]
        
        # Geodesic distance (more accurate for Lat/Lon)
        dist = geodesic((source_center_ll[1], source_center_ll[0]), 
                        (target_center_ll[1], target_center_ll[0])).meters
                        
        target_name = target_info.get('name') or f"Building_{target_info.get('osm_id')}"
        return f"{direction} of {target_name} ({int(dist)}m)"

    def merge_and_save_osm(self, pc_info: Dict[str, Any]):
        """
        Reads the OSM template JSON, updates it with generated scene graph data, 
        and saves strictly to self.output_json.
        """
        logger.info(f"Merging generated data into structure of {self.osm_template_file}...")
        
        # 1. Load Original Data (Template)
        with open(self.osm_template_file, 'r', encoding='utf-8') as f:
            osm_data = json.load(f)

        # 2. Build Lookup Table for O(1) Access
        # Maps osm_id string -> reference to the dictionary object inside osm_data
        lookup_table = {}
        target_keys = ['Polygon', 'Polyline', 'Point']
        
        for key in target_keys:
            if key in osm_data and isinstance(osm_data[key], list):
                for item in osm_data[key]:
                    # item structure: { "osm_id_123": { ... attributes ... } }
                    for osm_id, content in item.items():
                        lookup_table[str(osm_id)] = content

        # 3. Update Data In-Place using Lookup Table
        merged_count = 0
        for osm_id, gen_data in pc_info.items():
            if osm_id in lookup_table:
                target_dict = lookup_table[osm_id]
                
                # Update specific fields
                # Ensure keys match what is produced in run()
                target_dict.update({
                    'Landmark': gen_data.get('Landmark'),
                    '3D Spatial': gen_data.get('3D Spatial'), # Use space here to match JSON standard
                    'Topology': gen_data.get('Topology')
                })
                
                # Update height if available
                if gen_data.get('height'):
                    target_dict['height'] = gen_data['height']
                
                # Update image_caption if available
                if gen_data.get('image_caption'):
                    target_dict['image_caption'] = gen_data['image_caption']
                    
                merged_count += 1
        
        logger.info(f"Updated {merged_count} features in OSM data.")

        # 4. Save to the final output path
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(osm_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Final structured scene graph saved to {self.output_json}")

    def run(self, search_radius: float = 50.0):
        logger.info(f"Loading point cloud from {self.pc_path}")
        pcd = PyntCloud.from_file(str(self.pc_path))
        points_df = pcd.points
        
        # Load context layers
        env_layers = self.load_shapefiles(['roads', 'pois', 'water', 'transport'])
        buildings_gdf = self.load_shapefiles(['buildings']).get('buildings', gpd.GeoDataFrame())
        
        pc_info = {}
        unique_instances = points_df['instance'].unique()
        unique_instances = unique_instances[unique_instances != -1]
        
        logger.info(f"Processing {len(unique_instances)} instances...")
        
        # Store centroids for spatial search: ID -> (x_proj, y_proj)
        proj_centroids = {} 
        # Store lat/lon for JSON output and bearing calc: ID -> (lon, lat)
        ll_centroids = {} 

        # --- Phase 1: Attribute Extraction ---
        for ins_id in unique_instances:
            ins_str = str(int(ins_id))
            ins_points = points_df[points_df['instance'] == ins_id]
            
            # Default properties
            poly_geom = None
            props = {'osm_id': int(ins_id), 'name': 'unclassified', 'fclass': 'building'}
            
            # 1. Try matching with OSM Shapefile data
            if ins_str in self.osm_instance_map:
                map_info = self.osm_instance_map[ins_str]
                osm_id_val = map_info.get('osm_id')
                
                # Use osm_id_val as key for pc_info to match the OSM JSON structure
                key_id = str(osm_id_val)
                
                if not buildings_gdf.empty:
                    match = buildings_gdf[buildings_gdf['osm_id'].astype(str) == str(osm_id_val)]
                    if not match.empty:
                        row = match.iloc[0]
                        poly_geom = row.geometry
                        props.update({
                            'name': row.get('name', ''),
                            'fclass': row.get('fclass', ''),
                            'osm_id': osm_id_val
                        })
            else:
                # If not in map, skip or use instance ID? 
                # Assuming we only care about mapped objects for the final OSM JSON update
                key_id = ins_str 
            
            # 2. Fallback: Generate Convex Hull from points
            if poly_geom is None:
                pts = ins_points[['lon', 'lat']].values
                poly_geom = self.points_to_polygon(pts)
                props['name'] = f"Instance_{int(ins_id)}"
            
            # 3. Calculate Geometry Attributes
            centroid_proj = poly_geom.centroid
            
            # Reproject centroid back to Lat/Lon for output
            centroid_gdf = gpd.GeoDataFrame(geometry=[centroid_proj], crs=self.projected_crs)
            centroid_ll = centroid_gdf.to_crs(self.input_crs).geometry[0]
            
            height = self.get_height(ins_points['height'].values if 'height' in ins_points else ins_points['z'].values)
            area = int(poly_geom.area)
            
            # 4. Context Query
            Topology_desc = self.query_nearby_features(poly_geom, env_layers, radius=search_radius)
            
            # Store Data
            # Note: We store under 'key_id' (the OSM ID) to facilitate merging
            pc_info[key_id] = {
                'osm_id': props['osm_id'],
                'name': props['name'],
                'fclass': props['fclass'],
                'area': f"{area} m2",
                'height': f"{height} m",
                'center': [centroid_ll.x, centroid_ll.y],
                'Topology': Topology_desc,
                'Landmark': None, # Placeholder if logic exists later
                '3D Spatial': ""  # Placeholder
            }
            
            proj_centroids[key_id] = [centroid_proj.x, centroid_proj.y]
            ll_centroids[key_id] = [centroid_ll.x, centroid_ll.y]

        # --- Phase 2: Spatial Relation Graph ---
        logger.info("Computing spatial relations...")
        
        ids = list(proj_centroids.keys())
        coords_proj = np.array(list(proj_centroids.values()))
        
        if len(coords_proj) > 0:
            # Build KDTree using Metric coordinates (Meters)
            tree = cKDTree(coords_proj)
            
            for i, ins_id in enumerate(ids):
                # Query neighbors within 50 meters
                idx_list = tree.query_ball_point(coords_proj[i], r=search_radius)
                
                spatial_rels = []
                for neighbor_idx in idx_list:
                    neighbor_id = ids[neighbor_idx]
                    if neighbor_id == ins_id: continue
                    
                    target_info = pc_info[neighbor_id]
                    # Use Lat/Lon for Bearing and Geodesic distance
                    rel_desc = self.format_spatial_relation(
                        ll_centroids[ins_id], ll_centroids[neighbor_id], target_info
                    )
                    spatial_rels.append(rel_desc)
                
                # Update with the specific key "3D Spatial" to match merge function
                pc_info[ins_id]['3D Spatial'] = " | ".join(spatial_rels[:5])

        # --- Phase 3: Merge and Output ---
        # Directly call merge, no intermediate saving of pc_info
        self.merge_and_save_osm(pc_info)

def main():
    parser = argparse.ArgumentParser(description="Generate Scene Graph and Merge with OSM Structure.")
    
    parser.add_argument('--pc_file', type=str, required=True, help="Input annotated point cloud (.ply)")
    parser.add_argument('--shp_dir', type=str, required=True, help="OSM Shapefiles directory")
    parser.add_argument('--osm_map_file', type=str, default="osm_instance_map.json", help="OSM mapping file")

    # Important: Input Source
    parser.add_argument('--osm_json_file', type=str, required=True, help="Input: Raw OSM features JSON to be updated.")
    
    # Important: Final Output Destination
    parser.add_argument('--output_json', type=str, default="scene_structured.json", help="Output: Path for the final structured JSON.")
    
    parser.add_argument('--input_crs', type=int, default=4326, help="Input CRS EPSG code (WGS84)")
    parser.add_argument('--projected_crs', type=int, default=32649, help="Projected CRS EPSG code (Metric)")
    
    args = parser.parse_args()
    
    try:
        generator = SceneGraphGenerator(args)
        generator.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
    """
    python process/generate_scene_graph.py --pc_file D:\chen\dataset\guanggu2\target_rgb_labels_local_trans_anno.ply --shp_dir D:\chen\dataset\Guanggu\clip3_ori --osm_map_file "D:\chen\dataset\guanggu2\osm_instance_map.json" --osm_json_file "D:\chen\dataset\guanggu2\osm_features.json"  --output_json "D:\chen\dataset\guanggu2\scene_structured.json"
    """