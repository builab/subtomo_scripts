#!/usr/bin/env bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export variables
export SCRIPT_DIR


# Add script directory to PATH if needed
export PATH="$SCRIPT_DIR:$PATH"

echo "Environment set up. SCRIPT_DIR is $SCRIPT_DIR"