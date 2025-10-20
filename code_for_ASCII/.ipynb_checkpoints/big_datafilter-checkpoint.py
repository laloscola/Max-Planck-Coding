import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import sys

def chunk_datafilter(datafile, chunksize = 10_000_000, low_bound=0.1, high_bound=6):
    '''
    description: 
    - filters data according to specific ToT bounds
    - needs big_datacleaner.py processing first

    goal:
    - delete ToT's outside of range 
    - results in similar dataflow in both channels 

    thoughts: 
    - 
    '''

    try:
        print("Filtering data in chunks...")
        
        # Set up output filename
        base_name = os.path.basename(datafile)
        output_name = f"{base_name.replace('cleaned', 'filtered')}"
        
        # Set up chunk reader
        reader = pd.read_csv(datafile, sep='\t', skipinitialspace=True, chunksize=chunksize)

        for i, chunk in enumerate(reader):
            print(f"Filtering chunk {i+1}...")

            # Keep only rows where current is Ch 0 and next is Ch 1
            mask_pair = (chunk['Ch'] == 0.0) & (chunk['Ch'].shift(-1) == 1.0)   #this mask cancels all the Ch 1 data -> have to add following row later 
            
            # Keep only rows where both ToT_ns values are in range
            mask_range = (
                (chunk['ToT_ns'] > low_bound) & (chunk['ToT_ns'] < high_bound) &
                (chunk['ToT_ns'].shift(-1) > low_bound) & (chunk['ToT_ns'].shift(-1) < high_bound)
            )
            
            start_index = chunk.index[mask_pair & mask_range] #selects start row according to filtering
            keep_index = start_index.union(start_index + 1)  # union of start row and following row 
            
            # Final filtered DataFrame
            pairs = chunk.loc[keep_index]

            # Save filtered data in file 
            header = (i == 0)
            pairs.to_csv(output_name, sep='\t', index=False, mode='a', header=header)

        print(f" Cleaned data written to: {output_name}")

    except Exception as e:
        print(f" Error processing file: {e}")
        return None



if __name__ == "__main__":
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
        chunk_datafilter(datafile)
    else:
        print("No input data file provided")