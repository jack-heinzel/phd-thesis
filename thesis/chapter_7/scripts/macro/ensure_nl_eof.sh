#!/bin/bash

FILE="$1"

# Check if file ends with a newline
if [ -n "$(tail -c1 "$FILE")" ]; then
  echo >> "$FILE"
fi