import pandas as pd
import os
import sys

def big_datacleaner(datafile, chunksize=10_000_000):
    '''
    Filters common TDC data errors:
        - Acquisition mode: Common Start
        - Measurement mode: Lead_ToT (e.g. Lead_ToT11)
        - Uses chunking to handle large files without running out of memory.

    Solves:
        - First 8 lines are metadata and should be skipped
        - File is comma-separated (after AWK step)
        - Output is written as .txt using tab separator
        - '-' in 'ToT_ns' is treated as missing and converted to 0.0
        - Use 'float_format="%.4f"' to preserve float precision (optional)
    '''

    try:
        print("Cleaning data in chunks...")

        # Set up output filename
        base_name = os.path.basename(datafile)
        output_name = f"big_cleaned_{base_name.replace('_separated.txt', '.txt')}"
        
        # Set up chunk reader
        data_chunk = pd.read_csv(datafile, skiprows=8, skipinitialspace=True, chunksize=chunksize)

        for i, chunk in enumerate(data_chunk):
            print(f"Cleaning chunk {i+1}...")

            # 1. Replace '-' in ToT_ns with 0.0 and convert to float
            if 'ToT_ns' in chunk.columns:
                chunk['ToT_ns'] = chunk['ToT_ns'].replace('-', 0)
                chunk['ToT_ns'] = pd.to_numeric(chunk['ToT_ns'], errors='coerce')

            # 2. Replace NaNs in key columns with 0
            for col in ['Brd', 'Ch', 'deltaT_ns', 'ToT_ns']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].fillna(0)

            # 3. Drop rows where Ch == 0 and next row is also Ch == 0
            condition = (chunk['Ch'] == 0) & (chunk['Ch'].shift(-1) == 0)
            chunk_clean = chunk[~condition]

            # 4. Append to output file (no header after first chunk)
            header = (i == 0)
            chunk_clean.to_csv(output_name, sep='\t', index=False, mode='a', header=header)

        print(f" Cleaned data written to: {output_name}")

    except Exception as e:
        print(f" Error processing file: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
        chunk_datacleaner(datafile)
    else:
        print("No input data file provided")
