import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import os
import time

def csv_to_parquet(csv_file: str, parquet_file: str, chunksize: int = 10_000_000):
    """
    Convert Janus-generated CSV to Parquet in chunks.
    Handles encoding issues, bad rows, and mixed numeric types.
    Uses fixed schema to avoid schema mismatch.
    """
    col_names = ['Tstamp_us', 'Trg_Id', 'Entries', 'Board', 'Ch', 'ToA_ns', 'ToT_ns', 'Edge']

    try:
        with open(csv_file, 'r', encoding='latin1', errors='ignore') as f:
            clock_start = time.perf_counter()
            print('Starting CSV → Parquet conversion...')

            # Read comment/description block
            description_lines = []
            pos = f.tell()
            line = f.readline()
            while line and line.lstrip().startswith("//"):
                description_lines.append(line.rstrip("\n"))
                pos = f.tell()
                line = f.readline()
            f.seek(pos)

            # Define fixed schema with float64 for all numeric columns to avoid mismatch
            schema = pa.schema([
                ('Tstamp_us', pa.float64()),
                ('Trg_Id', pa.float64()),
                ('Entries', pa.float64()),
                ('Board', pa.float64()),
                ('Ch', pa.float64()),
                ('ToA_ns', pa.float64()),
                ('ToT_ns', pa.float64())
            ])

            writer = None

            for i, chunk in enumerate(pd.read_csv(
                f,
                chunksize=chunksize,
                header=0,
                names=col_names,
                encoding='latin1',
                on_bad_lines='skip',
                dtype=str
            )):
                # Convert numeric columns safely to float64
                for numeric_col in ['Tstamp_us', 'Trg_Id', 'Entries', 'Board', 'Ch', 'ToA_ns', 'ToT_ns']:
                    chunk[numeric_col] = pd.to_numeric(chunk[numeric_col], errors='coerce').astype('float64')

                # Drop 'Edge' column
                chunk = chunk.drop(columns=['Edge'], errors='ignore')

                # Convert dataframe to pyarrow Table with fixed schema
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)

                if writer is None:
                    meta = {b"file_description": "\n".join(description_lines).encode('utf-8')}
                    schema_with_meta = schema.with_metadata(meta)
                    writer = pq.ParquetWriter(parquet_file, schema_with_meta)

                print(f'Converting chunk {i+1}...')
                writer.write_table(table)

            if writer:
                writer.close()
                elapsed = time.perf_counter() - clock_start
                print(f'Finished writing Parquet file: {parquet_file}')
                print(f'Total time: {elapsed:.2f}s')

    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Janus CSV to Parquet in chunks")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("parquet_file", help="Path to output Parquet file")
    parser.add_argument("--chunksize", type=int, default=10_000_000, help="Rows per chunk")
    args = parser.parse_args()

    if os.path.exists(args.parquet_file):
        os.remove(args.parquet_file)

    csv_to_parquet(args.csv_file, args.parquet_file, args.chunksize)