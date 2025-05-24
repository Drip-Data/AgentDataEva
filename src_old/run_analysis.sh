#!/bin/bash
# OpenHands Data Processing Pipeline Runner

# Default values
INPUT_FILE="data/test.json"
OUTPUT_DIR="output"
VISUALIZE=false
DASHBOARD=false
WEB_DASHBOARD=false
PORT=12000

# Help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE       Path to the input JSON file (default: data/test.json)"
    echo "  -o, --output-dir DIR   Directory to save output files (default: output)"
    echo "  -v, --visualize        Generate visualizations"
    echo "  -d, --dashboard        Create a dashboard visualization"
    echo "  -w, --web-dashboard    Run the web dashboard"
    echo "  -p, --port PORT        Port to run the web dashboard on (default: 12000)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i data/test.json -o output -v -d"
    echo "  $0 --input data/test.json --web-dashboard --port 8080"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--visualize)
            VISUALIZE=true
            shift
            ;;
        -d|--dashboard)
            DASHBOARD=true
            shift
            ;;
        -w|--web-dashboard)
            WEB_DASHBOARD=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python main.py --input $INPUT_FILE --output-dir $OUTPUT_DIR"

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

if [ "$DASHBOARD" = true ]; then
    CMD="$CMD --dashboard"
fi

if [ "$WEB_DASHBOARD" = true ]; then
    CMD="$CMD --web-dashboard --port $PORT"
fi

# Run command
echo "Running: $CMD"
eval "$CMD"