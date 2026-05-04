#!/usr/bin/env python3

import json
from optparse import OptionParser

def round_floats(obj, precision=2):
    """Recursively round float values in JSON data."""
    if isinstance(obj, float):
        return "{:.2f}".format(round(obj, precision))
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(elem, precision) for elem in obj]
    else:
        return obj

def main():
    usage = "usage: %prog -i input.json -o output.json"
    parser = OptionParser(usage)
    parser.add_option("-i", "--input", dest="input_file",
                      help="Path to input JSON file")
    parser.add_option("-o", "--output", dest="output_file",
                      help="Path to output JSON file")

    (options, args) = parser.parse_args()

    if not options.input_file or not options.output_file:
        parser.error("Both input and output file paths are required.")

    # Load JSON
    with open(options.input_file, "r") as infile:
        data = json.load(infile)

    # Round floats
    rounded_data = round_floats(data)

    # Write result
    with open(options.output_file, "w") as outfile:
        json.dump(rounded_data, outfile, indent=2)

    print(f"Rounded JSON written to {options.output_file}")

if __name__ == "__main__":
    main()

