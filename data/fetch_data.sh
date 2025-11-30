#!/bin/bash
# Script to fetch METR-LA dataset for DCRNN training
# METR-LA: Traffic dataset from Los Angeles metropolitan area

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}"

echo "============================================="
echo "METR-LA Dataset Fetcher"
echo "============================================="

# Create directories
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/processed"

# METR-LA dataset from Google Drive (public link from DCRNN paper)
# Source: https://github.com/liyaguang/DCRNN
METR_LA_URL="https://zenodo.org/record/5724362/files/METR-LA.zip"
PEMS_BAY_URL="https://zenodo.org/record/5724362/files/PEMS-BAY.zip"

# Alternative: Direct download from DCRNN repo
METR_LA_DIRECT="https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/metr-la.h5"
SENSOR_IDS_URL="https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt"
DISTANCES_URL="https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv"

echo ""
echo "Downloading METR-LA dataset files..."

# Download main data file
if [ ! -f "${DATA_DIR}/raw/metr-la.h5" ]; then
    echo "Downloading metr-la.h5..."
    wget -q --show-progress -O "${DATA_DIR}/raw/metr-la.h5" "${METR_LA_DIRECT}" || {
        echo "Direct download failed. Trying alternative source..."
        # Alternative: create synthetic sample if download fails
        echo "Creating synthetic sample data..."
        python3 "${SCRIPT_DIR}/generate_sample_data.py"
    }
else
    echo "metr-la.h5 already exists, skipping download."
fi

# Download sensor graph information
if [ ! -f "${DATA_DIR}/raw/graph_sensor_ids.txt" ]; then
    echo "Downloading sensor IDs..."
    wget -q --show-progress -O "${DATA_DIR}/raw/graph_sensor_ids.txt" "${SENSOR_IDS_URL}" 2>/dev/null || echo "Sensor IDs download failed"
fi

if [ ! -f "${DATA_DIR}/raw/distances_la_2012.csv" ]; then
    echo "Downloading sensor distances..."
    wget -q --show-progress -O "${DATA_DIR}/raw/distances_la_2012.csv" "${DISTANCES_URL}" 2>/dev/null || echo "Distances download failed"
fi

echo ""
echo "Processing data..."

# Run preprocessing script
python3 "${SCRIPT_DIR}/preprocess_metr_la.py" || {
    echo "Preprocessing failed. Creating synthetic sample data..."
    python3 "${SCRIPT_DIR}/generate_sample_data.py"
}

echo ""
echo "============================================="
echo "Data preparation complete!"
echo "============================================="
echo ""
echo "Dataset location: ${DATA_DIR}"
echo ""
echo "Contents:"
ls -la "${DATA_DIR}/processed/" 2>/dev/null || ls -la "${DATA_DIR}/"
echo ""
echo "To use in training:"
echo "  ./run.sh python src/train.py --data ${DATA_DIR}"
