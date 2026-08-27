#!/bin/bash

################################################################################
# HYPERPARAMETERS - Configure these at the top
################################################################################

# API Key Configuration (for Gemini only)
# Option 1: Set here (not recommended for security)
# GEMINI_API_KEY="your-api-key-here"

# Option 2: Set as environment variable (recommended)
# export LANGEXTRACT_API_KEY="your-api-key-here"
# or
# export GOOGLE_API_KEY="your-api-key-here"

# Option 3: Load from .env file
# The script will automatically check for .env file in the repo root (/app in Docker)

# Model Configuration
MODEL_PROVIDER="gemini"  # Options: gemini, ollama, lmstudio
MODEL_NAME="gemini-3-flash-preview"  # For gemini: gemini-flash-latest, gemini-3.6-flash, etc.
TEMPERATURE=0.0

# Base directory detection (Docker vs local)
if [ -d "/app/kgb" ]; then
    BASE_DIR="/app"
    PYTHON="python3"
else
    BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# Use repo venv when available, otherwise prefer an explicit 3.11 interpreter.
if [ -x "${BASE_DIR}/.venv/bin/python" ]; then
    PYTHON="${BASE_DIR}/.venv/bin/python"
else
    PYTHON=""
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    echo "ERROR: Python 3.11+ is required. Selected interpreter: $PYTHON"
    [ -n "$PYTHON" ] && "$PYTHON" --version 2>/dev/null || true
    exit 1
fi

# Input Data
INPUT_FILE="${BASE_DIR}/data/legal/legal_background.jsonl"  # Path to your input file
TEXT_FIELD="text"  # Field name containing the text to analyze
ID_FIELD="id"  # Field name containing record IDs
RECORD_IDS="UKSC-2009-0143"  # Specific record ID(s) to process (comma-separated, or empty for all)

# Domain Configuration
DOMAIN="legal"  # Use: kgb list domains

# Extraction Mode
MODE="open"  # Options: open, constrained

# Output Configuration
OUTPUT_DIR="${BASE_DIR}/test_outputs/single_extraction_$(date +%Y%m%d_%H%M%S)"

# Connectivity Augmentation Configuration
MAX_DISCONNECTED=1  # Maximum acceptable disconnected components
MAX_ITERATIONS=5    # Max refinement iterations

# Visualization Options
CREATE_NETWORK_VIZ=true  # Create NetworkX/Plotly graph visualization
CREATE_EXTRACTION_VIZ=true  # Create langextract HTML visualization
DARK_MODE=false  # Enable dark mode for network visualization
LAYOUT="spring"  # Graph layout (spring, circular, kamada_kawai, shell)
GROUP_BY="entity_type"  # Options: entity_type, relation

# Processing Options
MAX_WORKERS=""  # Leave empty for default, or set number
TIMEOUT=120  # Request timeout in seconds

################################################################################
# Script Execution - Do not modify below this line
################################################################################

set -e  # Exit on error

# Load .env file if it exists
if [ -f "${BASE_DIR}/.env" ]; then
    echo "Loading environment variables from ${BASE_DIR}/.env"
    export $(grep -v '^#' "${BASE_DIR}/.env" | xargs)
fi

# Check API key for Gemini
if [ "$MODEL_PROVIDER" = "gemini" ]; then
    if [ -z "$LANGEXTRACT_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
        echo "================================================================================"
        echo "ERROR: Gemini API key not found!"
        echo "================================================================================"
        echo ""
        echo "Please set your API key using one of these methods:"
        echo ""
        echo "1. Environment variable (recommended):"
        echo "   export LANGEXTRACT_API_KEY='your-api-key-here'"
        echo "   or"
        echo "   export GOOGLE_API_KEY='your-api-key-here'"
        echo ""
        echo "2. Create .env file:"
        echo "   echo 'LANGEXTRACT_API_KEY=your-api-key-here' > ${BASE_DIR}/.env"
        echo ""
        echo "3. Set in script (not recommended):"
        echo "   Edit this script and uncomment the GEMINI_API_KEY line at the top"
        echo ""
        echo "Get your API key from: https://aistudio.google.com/app/apikey"
        echo "================================================================================"
        exit 1
    fi

    # Use GEMINI_API_KEY if set in script, otherwise use env vars
    if [ -n "$GEMINI_API_KEY" ]; then
        export LANGEXTRACT_API_KEY="$GEMINI_API_KEY"
    fi
fi

echo "================================================================================"
echo "KNOWLEDGE GRAPH EXTRACTION - SINGLE TEXT TEST"
echo "Using Typer CLI for unified command-line interface"
echo "================================================================================"
echo "Model Provider: $MODEL_PROVIDER"
echo "Model Name: $MODEL_NAME"
echo "Input File: $INPUT_FILE"
echo "Record IDs: $RECORD_IDS"
echo "Domain: $DOMAIN"
echo "Mode: $MODE"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Connectivity Configuration:"
echo "  • Max disconnected: $MAX_DISCONNECTED"
echo "  • Max iterations: $MAX_ITERATIONS"
if [ "$MODEL_PROVIDER" = "gemini" ]; then
    if [ -n "$LANGEXTRACT_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ]; then
        echo "API Key: ✓ Configured"
    fi
fi
echo "================================================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save run metadata
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$OUTPUT_DIR/metadata.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "model_provider": "$MODEL_PROVIDER",
  "model_name": "$MODEL_NAME",
  "temperature": $TEMPERATURE,
  "domain": "$DOMAIN",
  "mode": "$MODE",
  "input_file": "$INPUT_FILE",
  "text_field": "$TEXT_FIELD",
  "id_field": "$ID_FIELD",
  "record_ids": "$RECORD_IDS",
  "max_disconnected": $MAX_DISCONNECTED,
  "max_iterations": $MAX_ITERATIONS,
  "max_workers": "${MAX_WORKERS:-null}",
  "timeout": $TIMEOUT,
  "create_network_viz": $CREATE_NETWORK_VIZ,
  "create_extraction_viz": $CREATE_EXTRACTION_VIZ,
  "dark_mode": $DARK_MODE,
  "layout": "$LAYOUT",
  "group_by": "$GROUP_BY"
}
EOF

# Build common CLI options
CLI_OPTS=""
CLI_OPTS="$CLI_OPTS --input $INPUT_FILE"
CLI_OPTS="$CLI_OPTS --output-dir $OUTPUT_DIR"
CLI_OPTS="$CLI_OPTS --domain $DOMAIN"
CLI_OPTS="$CLI_OPTS --mode $MODE"
CLI_OPTS="$CLI_OPTS --client $MODEL_PROVIDER"
CLI_OPTS="$CLI_OPTS --text-field $TEXT_FIELD"
CLI_OPTS="$CLI_OPTS --id-field $ID_FIELD"
CLI_OPTS="$CLI_OPTS --record-ids $RECORD_IDS"
CLI_OPTS="$CLI_OPTS --temp $TEMPERATURE"
CLI_OPTS="$CLI_OPTS --timeout $TIMEOUT"

if [ -n "$MODEL_NAME" ]; then
    CLI_OPTS="$CLI_OPTS --model $MODEL_NAME"
fi

if [ -n "$MAX_WORKERS" ]; then
    CLI_OPTS="$CLI_OPTS --workers $MAX_WORKERS"
fi


# ================================================================================
# STEP 1: EXTRACT TRIPLES
# ================================================================================
echo ""
echo "================================================================================"
echo "STEP 1: EXTRACTING TRIPLES"
echo "================================================================================"

if ! $PYTHON -m kgb extract $CLI_OPTS; then
    echo "ERROR: Extraction failed"
    exit 1
fi

# ================================================================================
# STEP 2: AUGMENT CONNECTIVITY
# ================================================================================
echo ""
echo "================================================================================"
echo "STEP 2: AUGMENTING CONNECTIVITY"
echo "================================================================================"

if ! $PYTHON -m kgb augment connectivity $CLI_OPTS \
    --max-disconnected $MAX_DISCONNECTED \
    --max-iterations $MAX_ITERATIONS; then
    echo "ERROR: Augmentation failed"
    exit 1
fi

# ================================================================================
# STEP 3: CONVERT TO GRAPHML
# ================================================================================
echo ""
echo "================================================================================"
echo "STEP 3: CONVERTING TO GRAPHML"
echo "================================================================================"

JSON_DIR="$OUTPUT_DIR/extracted_json"
GRAPHML_DIR="$OUTPUT_DIR/graphml"

if ! $PYTHON -m kgb convert --input "$JSON_DIR" --output "$GRAPHML_DIR"; then
    echo "ERROR: Conversion failed"
    exit 1
fi

# ================================================================================
# STEP 4: CREATE VISUALIZATIONS
# ================================================================================

# Network visualization
if [ "$CREATE_NETWORK_VIZ" = true ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 4a: CREATING NETWORK VISUALIZATION"
    echo "================================================================================"

    NETWORK_VIZ_DIR="$OUTPUT_DIR/network_viz"
    VIZ_OPTS="--input $GRAPHML_DIR --output $NETWORK_VIZ_DIR --layout $LAYOUT"
    if [ "$DARK_MODE" = true ]; then
        VIZ_OPTS="$VIZ_OPTS --dark-mode"
    fi

    if ! $PYTHON -m kgb visualize network $VIZ_OPTS; then
        echo "WARNING: Network visualization failed"
    fi
fi

# Extraction visualization
if [ "$CREATE_EXTRACTION_VIZ" = true ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 4b: CREATING EXTRACTION VISUALIZATION"
    echo "================================================================================"

    EXTRACTION_VIZ_DIR="$OUTPUT_DIR/extraction_viz"
    
    if ! $PYTHON -m kgb visualize extraction \
        --input "$INPUT_FILE" \
        --triples "$JSON_DIR" \
        --output "$EXTRACTION_VIZ_DIR" \
        --text-field "$TEXT_FIELD" \
        --id-field "$ID_FIELD" \
        --group-by "$GROUP_BY"; then
        echo "WARNING: Extraction visualization failed"
    fi
fi

# ================================================================================
# SUMMARY
# ================================================================================
echo ""
echo "================================================================================"
echo "SUCCESS! All files generated in: $OUTPUT_DIR"
echo "================================================================================"
echo ""
echo "View the results:"
echo "  JSON triples: $JSON_DIR/"
echo "  GraphML: $GRAPHML_DIR/"
if [ "$CREATE_NETWORK_VIZ" = true ]; then
    echo "  Network visualization: file://$OUTPUT_DIR/network_viz/"
fi
if [ "$CREATE_EXTRACTION_VIZ" = true ]; then
    echo "  Extraction visualization: file://$OUTPUT_DIR/extraction_viz/"
fi
echo ""
echo "Output structure:"
tree -L 2 "$OUTPUT_DIR" 2>/dev/null || ls -R "$OUTPUT_DIR"
echo ""
echo "================================================================================"
echo "To explore available commands, run:"
echo "  kgb --help"
echo "  kgb list domains"
echo "  kgb list clients"
echo "================================================================================"
