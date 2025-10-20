import pandas as pd
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time

def csv_to_parquet(csv_file : str, parquet_file : str, chunksize=10_000_000):
    '''This converter works only for CSV files that are given by the Janus software.

    Remarks:
    - error in Janus software gives an extra datacolumn called 'Edge' that doesn't have data 
    - the names of the CSV files are also changed to match the names of the ASCII files:
        - there is an extra column 'Entries'
    '''

    try:
        # Split description and stream the data in chunks
        with open(csv_file, "r", encoding='utf-8') as f:

            clock_start = time.perf_counter()

            print('Starting conversion from CSV to Parquet...')
            
            # Read and store comment block
            description_lines = []
            pos = f.tell()
            line = f.readline()
            while line and line.lstrip().startswith("//"):
                description_lines.append(line.rstrip("\n"))
                pos = f.tell()
                line = f.readline()
            f.seek(pos)  # rewind to header row
            

            # initialize parquet writer 
            writer = None
            col_names = ['Tstamp_us', 'Trg_Id', 'Entries', 'Board', 'Ch', 'ToA_ns', 'ToT_ns','Edge'] # adjusts column names and drops 'Edge'
            for i, chunk in enumerate(pd.read_csv(f, chunksize=chunksize, header=0, names=col_names)):  # chunked iteration
                table = pa.Table.from_pandas(chunk.drop(columns=['Edge']), preserve_index=False)
                print(f'Converting chunk: {i+1}...')
                
                if writer is None:
                    # Attach metadata only once on first write
                    meta = dict(table.schema.metadata or {})
                    meta[b"file_description"] = "\n".join(description_lines).encode("utf-8")  
                    table = table.replace_schema_metadata(meta)
                    writer = pq.ParquetWriter(parquet_file, table.schema)  
                writer.write_table(table)
                
            if writer is not None:
                writer.close()
                time_diff = time.perf_counter() - clock_start
                print(f'Finished writing parquet file to: {parquet_file}')
                print(f'Total time to run code is: {time_diff}s')
    
    except Exception as e:
        print(f"Error: {e}")
        return None
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert large CSV to Parquet in chunks")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("parquet_file", help="Path to output Parquet file")
    parser.add_argument("--chunksize", type=int, default=10000000, help="Number of rows per chunk (default: 10,000,000)")
    args = parser.parse_args()

    if os.path.exists(args.parquet_file):
        os.remove(args.parquet_file)  # Avoid appending to old file

    csv_to_parquet(args.csv_file, args.parquet_file, args.chunksize)

