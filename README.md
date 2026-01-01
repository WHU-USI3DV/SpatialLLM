# SpatialLLM：Enhancing Large Language Models for Urban Spatial Understanding

## 📋 Overview

![overview](overview.png)

<p align="justify">
SpatialLLM is a comprehensive framework for enhancing Large Language Models with urban spatial understanding capabilities. This project integrates point cloud processing, OpenStreetMap (OSM) data, and multi-view images to structured scene text, enabling LLMs to perform complex spatial reasoning tasks in urban environments.
</p>

## 🚀 install

```bash
# Create and activate conda environment
conda create -n spatialllm python=3.8
conda activate spatialllm

# Install geospatial dependencies via conda
conda install -c conda-forge geopandas
conda install -c conda-forge geopy
conda install -c conda-forge gdal

# Install remaining dependencies via pip
pip install -r requirements.txt
```

## 🔧 Data Processing Pipeline

### Step 1: Extract OSM Information

```bash
python process/osm_process.py \
    --shp_path Directory containing OSM shapefiles (buildings, roads, etc.) \
    --output Output JSON file path
```

### Step 2: Automatic Point Cloud Annotation

```bash
python process/auto_annotate.py \
    --work_dir Path containing all input/output files \
    --shp_dir Directory containing OSM shapefiles \
    --pc_file Point cloud filename (`.ply` or `.txt`) in `work_dir` \
    --control_txt Control text file for annotation parameters \
    --res Raster resolution

# Control Points File
# Format: `x y z lon lat` (one point per line)
```

### Step 3: Generate Scene Graph

```bash
python process/generate_scene_graph.py \
    --pc_file Annotated point cloud from Step 2 \
    --shp_dir OSM shapefiles directory \
    --osm_map_file Instance mapping file from Step 2 \
    --osm_json_file **INPUT** OSM features to be enriched \
    --output_json **OUTPUT** Final structured scene graph
```

## Infer

```bash
python infer/inference.py 
```

> 💡 **Tip:** You can also directly use the structured scene text as context in web-based LLM interfaces (ChatGPT, Claude, etc.) for interactive spatial reasoning conversations.

## 📚 Examples

We provide multiple examples demonstrating various spatial understanding capabilities. Please refer to the `examples/` directory for detailed results.


# 🤝 Acknowledgement

SpatialLLM is built upon the extremely wonderful [UrbanBIS](https://github.com/fullcyxuc/B-Seg)

# Contact us

If you find this repo helpful, please give us a star. For any questions, please contact us via chenjb67@whu.edu.cn.
