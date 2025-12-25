import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from osgeo import gdal, osr
from pyntcloud import PyntCloud
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PointCloudAnnotator:
    def __init__(self, args):
        self.work_dir = Path(args.work_dir)
        self.shp_dir = Path(args.shp_dir)
        self.pc_filename = args.pc_file
        self.control_txt = self.work_dir / args.control_txt
        self.res = args.res
        self.projected_crs = args.projected_crs
        
        if not self.work_dir.exists():
            raise FileNotFoundError(f"Working directory not found: {self.work_dir}")
            
        self.pc_path = self.work_dir / self.pc_filename
        
        # Bounding box and grid state
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.grid_points = {} # Dict mapping (grid_y, grid_x) -> list of point indices

    def load_pc(self, file_path: Path) -> np.ndarray:
        """Loads point cloud data (.txt or .ply) into a numpy array."""
        logger.info(f"Loading point cloud: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            data = np.loadtxt(file_path)
            data = data[:, :6]  # xyz rgb
        elif file_extension == '.ply':
            pcd = PyntCloud.from_file(str(file_path))
            xyz = pcd.points[["x", "y", "z"]].to_numpy()
            rgb = pcd.points[["red", "green", "blue"]].to_numpy().astype(int)
            
            # Check for existing extra attributes
            extras = []
            for col in ['lon', 'lat', 'height']:
                if col in pcd.points.columns:
                    extras.append(pcd.points[[col]].to_numpy())
            
            if extras:
                data = np.hstack([xyz, rgb] + extras)
            else:
                data = np.hstack([xyz, rgb])
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        self.x_min, self.y_min, _ = np.min(data[:, :3], axis=0)
        self.x_max, self.y_max, _ = np.max(data[:, :3], axis=0)
        
        logger.info(f"Point cloud loaded. Shape: {data.shape}")
        return data

    def load_control_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads control points and normalizes to pixel coordinates."""
        if not self.control_txt.exists():
            raise FileNotFoundError(f"Control points file not found: {self.control_txt}")
            
        logger.info(f"Loading control points from {self.control_txt}")
        # Expected format in file: pixel_x, pixel_y, ..., lon, lat
        raw_data = np.loadtxt(self.control_txt, delimiter=',')
        
        c_pc = raw_data[:, :2]   # Original PC coordinates
        c_map = raw_data[:, -2:] # Map coordinates (Lon, Lat)

        # Normalize to pixel coordinates relative to the bounding box
        c_pixel = np.zeros_like(c_pc)
        c_pixel[:, 0] = (c_pc[:, 0] - self.x_min) / self.res
        c_pixel[:, 1] = (c_pc[:, 1] - self.y_min) / self.res

        return c_pixel, c_map

    def grid_pc(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rasterizes the point cloud into a top-down view using vectorization."""
        logger.info("Rasterizing point cloud to grid...")
        pc = data[:, :3]
        colors = data[:, 3:6]

        self.x_min, self.y_min, _ = np.min(pc, axis=0)
        self.x_max, self.y_max, _ = np.max(pc, axis=0)
        
        x_range = int((self.x_max - self.x_min) / self.res) + 1
        y_range = int((self.y_max - self.y_min) / self.res) + 1
        
        # Vectorized grid calculation
        grid_x = ((pc[:, 0] - self.x_min) / self.res).astype(int)
        grid_y = ((pc[:, 1] - self.y_min) / self.res).astype(int)
        
        # Filter points within bounds
        valid_mask = (grid_x >= 0) & (grid_x < x_range) & (grid_y >= 0) & (grid_y < y_range)
        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        pc_valid = pc[valid_mask]
        colors_valid = colors[valid_mask]
        indices = np.where(valid_mask)[0]
        
        # Create a DataFrame for efficient grouping
        df = pd.DataFrame({
            'y': grid_y, 
            'x': grid_x, 
            'z': pc_valid[:, 2], 
            'idx': indices,
            'r': colors_valid[:, 0],
            'g': colors_valid[:, 1],
            'b': colors_valid[:, 2]
        })
        
        # 1. Populate grid_points for labeling later (mapping pixel -> list of point indices)
        # Group by pixel and aggregate indices into lists
        self.grid_points = df.groupby(['y', 'x'])['idx'].apply(list).to_dict()
        
        # 2. Create the image (Height map and Color map)
        # Sort by Z (ascending) and drop duplicates keeping the last (highest) point
        df_top = df.sort_values('z').drop_duplicates(subset=['y', 'x'], keep='last')
        
        height_grid = np.full((y_range, x_range), -np.inf)
        color_grid = np.zeros((y_range, x_range, 3), dtype=np.uint8)
        
        # Assign values using numpy indexing
        rows = df_top['y'].values
        cols = df_top['x'].values
        
        height_grid[rows, cols] = df_top['z'].values
        color_grid[rows, cols, 0] = df_top['r'].values
        color_grid[rows, cols, 1] = df_top['g'].values
        color_grid[rows, cols, 2] = df_top['b'].values
        
        return height_grid, color_grid

    def calculate_height_scale(self, min_dist_threshold: float = 10.0) -> float:
        """Estimates Z-axis scale factor using Haversine distance vs Point Cloud distance."""
        try:
            raw_data = np.loadtxt(self.control_txt, delimiter=',')
        except Exception:
            return 1.0

        if len(raw_data) < 2: 
            return 1.0

        pc_xy = raw_data[:, :2] 
        map_ll = raw_data[:, -2:] 
        
        def haversine(lon1, lat1, lon2, lat2):
            R = 6371000 
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
            return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        scales = []
        for i in range(len(pc_xy)):
            for j in range(i + 1, len(pc_xy)):
                pc_dist = np.linalg.norm(pc_xy[i] - pc_xy[j])
                geo_dist = haversine(map_ll[i, 0], map_ll[i, 1], map_ll[j, 0], map_ll[j, 1])
                
                if geo_dist > min_dist_threshold and pc_dist > 0:
                    scales.append(geo_dist / pc_dist)
                    
        return np.mean(scales) if scales else 1.0

    def register_pc(self) -> Tuple[str, str]:
        """Performs affine registration and saves the projected GeoTIFF."""
        logger.info("Starting Registration...")
        
        data = self.load_pc(self.pc_path)
        _, color_grid = self.grid_pc(data)
        
        # Calculate Affine Transform Matrix
        pixel_coords, map_coords = self.load_control_points()
        
        # Least Squares: A * X = B
        # A: [pixel_x, pixel_y, 1]
        A = np.column_stack([pixel_coords, np.ones(len(pixel_coords))])
        B = map_coords
        
        X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        
        # Construct GDAL GeoTransform
        # X matrix mapping: x_geo = X[0,0]*x + X[1,0]*y + X[2,0]
        geotransform = (X[2, 0], X[0, 0], X[1, 0], X[2, 1], X[0, 1], X[1, 1])
        
        # Save GeoTIFF
        output_tif = self.work_dir / "raster_proj.tif"
        rows, cols, bands = color_grid.shape
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(str(output_tif), cols, rows, bands, gdal.GDT_Byte)
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326) # WGS84
        out_ds.SetProjection(srs.ExportToWkt())
        out_ds.SetGeoTransform(geotransform)
        
        for i in range(bands):
            out_ds.GetRasterBand(i + 1).WriteArray(color_grid[:, :, i])
        out_ds = None
        logger.info(f"Saved GeoTIFF to {output_tif}")
        
        # Transform Point Cloud Coordinates
        # 1. Pixel coordinates
        pc_xy_pixel = data[:, :2].copy()
        pc_xy_pixel[:, 0] = (pc_xy_pixel[:, 0] - self.x_min) / self.res
        pc_xy_pixel[:, 1] = (pc_xy_pixel[:, 1] - self.y_min) / self.res
        
        # 2. Affine Transform (Lat/Lon)
        # Vectorized transform: [x, y, 1] @ X
        ones = np.ones((len(pc_xy_pixel), 1))
        transformed_xy = np.hstack([pc_xy_pixel, ones]) @ X
        
        # 3. Z Scale
        z_scale_factor = self.calculate_height_scale()
        transformed_z = data[:, 2] * z_scale_factor
        
        # Save Transformed PLY
        output_ply = self.work_dir / f"{self.pc_path.stem}_trans.ply"
        
        new_data = np.column_stack([
            data[:, :3],    # Original XYZ
            data[:, 3:6],   # RGB
            transformed_xy, # Lon, Lat
            transformed_z   # Scaled Height
        ])
        
        columns = ['x', 'y', 'z', 'red', 'green', 'blue', 'lon', 'lat', 'height']
        df = pd.DataFrame(new_data, columns=columns)
        
        # Optimize types
        for col in ['red', 'green', 'blue']:
            df[col] = df[col].astype(np.uint8)
            
        PyntCloud(df).to_file(str(output_ply))
        logger.info(f"Saved Transformed Point Cloud to {output_ply}")
        
        # Save registration metadata
        with open(self.work_dir / "registration_params.txt", 'w') as f:
            f.write(f"GeoTransform: {geotransform}\n")
            f.write(f"Z Scale: {z_scale_factor}\n")
            
        return str(output_ply), str(output_tif)

    def label_pc(self, trans_ply_path: str, tif_path: str, layers: List[str] = None):
        """Annotates the point cloud using OSM shapefiles."""
        if layers is None:
            layers = ['buildings', 'landuse', 'pois', 'roads']
            
        logger.info("Starting Auto-Labeling...")
        
        # Load PC Data
        pcd = PyntCloud.from_file(trans_ply_path)
        original_df = pcd.points
        
        # Ensure grid is populated (in case this method is called standalone)
        if not self.grid_points:
             # Reconstruct grid using original XYZ columns (first 3)
            self.grid_pc(original_df[['x', 'y', 'z', 'red', 'green', 'blue']].to_numpy())

        with rasterio.open(tif_path) as src:
            transform = src.transform
            h, w = src.shape
            
        num_points = len(original_df)
        labels = np.full(num_points, -1, dtype=np.int32)
        instances = np.full(num_points, -1, dtype=np.int32)
        
        label_map = {} # {id: class_name}
        class_to_id = {} # {class_name: id}
        osm_instance_map = {} # {instance_id: {info}}
        
        curr_label_id = 1
        curr_instance_id = 1
        
        for layer_name in layers:
            shp_file = self.shp_dir / f"{layer_name}.shp"
            if not shp_file.exists():
                logger.warning(f"Layer {layer_name} not found, skipping.")
                continue
                
            logger.info(f"Processing layer: {layer_name}")
            gdf = gpd.read_file(shp_file)
            
            # Unify CRS to WGS84
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            
            # Sort by area (projected) to handle overlaps (smaller features on top)
            gdf_proj = gdf.to_crs(epsg=self.projected_crs)
            gdf['area_m2'] = gdf_proj.geometry.area
            gdf = gdf.sort_values('area_m2', ascending=False)
            
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.is_empty: continue
                
                try:
                    # Create raster mask
                    mask = geometry_mask([geom], out_shape=(h, w), transform=transform, invert=True)
                except Exception as e:
                    logger.error(f"Mask error ID {row.get('osm_id')}: {e}")
                    continue
                
                # Get pixel coordinates inside the mask
                rows_idx, cols_idx = np.where(mask)
                
                # Retrieve point indices from the pre-computed grid
                valid_point_indices = []
                for r, c in zip(rows_idx, cols_idx):
                    # Direct dictionary lookup is O(1)
                    if (r, c) in self.grid_points:
                        valid_point_indices.extend(self.grid_points[(r, c)])
                        
                if not valid_point_indices:
                    continue
                    
                # Handle labeling
                fclass = row.get('fclass', 'unknown')
                if fclass not in class_to_id:
                    class_to_id[fclass] = curr_label_id
                    label_map[curr_label_id] = fclass
                    curr_label_id += 1
                
                l_id = class_to_id[fclass]
                
                # Assign labels
                labels[valid_point_indices] = l_id
                instances[valid_point_indices] = curr_instance_id
                
                osm_instance_map[str(curr_instance_id)] = {
                    'shp_name': layer_name,
                    'osm_id': str(row.get('osm_id', idx)),
                    'fclass': fclass
                }
                curr_instance_id += 1
                
        # Save Final Output
        output_ply = self.work_dir / f"{Path(trans_ply_path).stem}_anno.ply"
        
        original_df['label'] = labels
        original_df['instance'] = instances
        
        PyntCloud(original_df).to_file(str(output_ply))
        
        # Save Metadata
        with open(self.work_dir / "label_map.json", 'w') as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=4)
            
        with open(self.work_dir / "osm_instance_map.json", 'w') as f:
            json.dump(osm_instance_map, f, indent=4)
            
        logger.info(f"Annotation done. Saved to {output_ply}")

def main():
    parser = argparse.ArgumentParser(description="Auto Register and Label Point Cloud using OSM data.")
    
    parser.add_argument('--work_dir', type=str, required=True, help="Working directory path.")
    parser.add_argument('--shp_dir', type=str, required=True, help="OSM Shapefiles directory.")
    parser.add_argument('--pc_file', type=str, default="fused.ply", help="Input Point Cloud filename.")
    parser.add_argument('--control_txt', type=str, default="control_points.txt", help="Control points filename.")
    parser.add_argument('--res', type=float, default=0.5, help="Grid resolution (meters/pixel).")
    parser.add_argument('--projected_crs', type=int, default=32649, help="EPSG code for area calculation.")
    
    args = parser.parse_args()
    
    annotator = PointCloudAnnotator(args)
    trans_ply, tif_path = annotator.register_pc()
    annotator.label_pc(trans_ply, tif_path)

if __name__ == '__main__':
    main()
    """
    python process/auto_annotate.py   --work_dir D:\chen\dataset\guanggu2  --shp_dir D:\chen\dataset\Guanggu\clip3_ori  --pc_file target_rgb_labels_local.txt  --res 0.5
    """