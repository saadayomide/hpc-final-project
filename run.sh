#!/bin/bash
# Wrapper script for building and running the Apptainer container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="project.sif"
CONTAINER_PATH="${SCRIPT_DIR}/env/${CONTAINER_NAME}"
DEF_FILE="${SCRIPT_DIR}/env/project.def"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to build container
build_container() {
    echo -e "${YELLOW}Building Apptainer container...${NC}"
    cd "${SCRIPT_DIR}/env"
    
    if [ -f "${CONTAINER_NAME}" ]; then
        echo -e "${YELLOW}Container already exists. Use --force to rebuild.${NC}"
        if [ "$1" != "--force" ]; then
            return 0
        fi
        rm -f "${CONTAINER_NAME}"
    fi
    
    apptainer build "${CONTAINER_NAME}" "${DEF_FILE}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Container built successfully!${NC}"
    else
        echo -e "${RED}Container build failed!${NC}"
        exit 1
    fi
}

# Function to run container
run_container() {
    if [ ! -f "${CONTAINER_PATH}" ]; then
        echo -e "${YELLOW}Container not found. Building...${NC}"
        build_container
    fi
    
    echo -e "${GREEN}Running container with command: $@${NC}"
    
    # Mount necessary directories
    apptainer exec --nv \
        --bind "${SCRIPT_DIR}:/workspace" \
        --pwd /workspace \
        "${CONTAINER_PATH}" \
        "$@"
}

# Main script logic
case "${1:-}" in
    build|--build)
        build_container "${2:-}"
        ;;
    *)
        run_container "$@"
        ;;
esac

