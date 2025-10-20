import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def TDC_ToT_histograms(datafile, chunksize=10_000_000, num_bins=135, tot_range=(1.7, 3.2), delta_range=(0, 250)):
    """ Plots ToT histograms of both channels in chunks (memory safe).
    """
    try:

        '''###overlapping histogram plot if needed
        plt.hist(data0['ToT_ns'], bins=num_bins, alpha=0.5, range=(0.1,3.4), label= 'Channel 0 data')
        plt.hist(data1['ToT_ns'], bins=num_bins, alpha=0.5, range=(0.1,3.4), label= 'Channel 1 data')
        plt.title(r"ToT in $ns$ of both channel for first 10 Million entries")
        plt.xlabel(r"ToT in $ns$")
        plt.ylabel(r"frequency")
        plt.legend()
        plt.show()
        '''

        ###plt style
        plt.style.use('ggplot')
        plt.rcParams['text.usetex']=False
        
        # ToT bin edges
        bin_edges_tot = np.linspace(tot_range[0], tot_range[1], num_bins + 1)
        hist0_tot = np.zeros(num_bins, dtype=int)
        hist1_tot = np.zeros(num_bins, dtype=int)

        # deltaT bin edges 
        bin_edges_delta = np.linspace(delta_range[0], delta_range[1], num_bins + 1)
        hist0_delta= np.zeros(num_bins, dtype=int)
        hist1_delta = np.zeros(num_bins, dtype=int)
        
        column_names = ['Tstamp_us', 'Trg_Id', 'Brd', 'Ch', 'deltaT_ns', 'ToT_ns']

        print("Accumulating histograms in chunks...")
        reader = pd.read_csv(datafile, comment='/', names=column_names, skipinitialspace=True, chunksize=chunksize)

        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i+1}...")

            
            # Split by channel
            data0_tot = chunk[chunk["Ch"] == 0]['ToT_ns'].dropna()
            data1_tot = chunk[chunk["Ch"] == 1]['ToT_ns'].dropna()
            data0_delta = chunk[chunk["Ch"] == 0]['deltaT_ns'].dropna()
            data1_delta = chunk[chunk["Ch"] == 1]['deltaT_ns'].dropna()
            
            # Update ToT histograms
            h0_tot, _ = np.histogram(data0_tot, bins=bin_edges_tot)
            h1_tot, _ = np.histogram(data1_tot, bins=bin_edges_tot)
            hist0_tot += h0_tot
            hist1_tot += h1_tot

            # Update deltaT histograms
            h0_delta, _ = np.histogram(data0_delta, bins=bin_edges_delta)
            h1_delta, _ = np.histogram(data1_delta, bins=bin_edges_delta)
            hist0_delta += h0_delta
            hist1_delta += h1_delta

            

        # Plot results
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        # ToT Histogram for Channel 0
        axs[0, 0].bar(bin_edges_tot[:-1], hist0_tot, width=np.diff(bin_edges_tot), align="edge", alpha=0.7, color="blue")
        axs[0, 0].set_title("Channel 0")
        axs[0, 0].set_xlabel("ToT in ns")
        axs[0, 0].set_ylabel("Frequency")

        # ToT Histogram for Channel 1
        axs[0, 1].bar(bin_edges_tot[:-1], hist1_tot, width=np.diff(bin_edges_tot), align="edge", alpha=0.7, color="red")
        axs[0, 1].set_title("Channel 1")
        axs[0, 1].set_xlabel("ToT in ns")

        # deltaT Histogram for Channel 0 
        axs[1,0].bar(bin_edges_delta[:-1], hist0_delta, width=np.diff(bin_edges_delta), align='edge', alpha=0.7, color='blue')
        axs[1,0].set_xlabel(r'$\Delta T$ in ns')
        axs[1,0].set_ylabel('Frequency')
        # deltaT Histogram for Channel 1 
        axs[1,1].bar(bin_edges_delta[:-1], hist1_delta, width=np.diff(bin_edges_delta), align='edge', alpha=0.7, color='red')
        axs[1,1].set_xlabel(r'$\Delta T$ in ns')
        
        # Seaborn KDE (approximate, sample only a subset to save memory)
        sample_rows = 1_000_000
        sample_data = pd.read_csv(datafile, delimiter="\t", nrows=sample_rows)
        data0_sample = sample_data[sample_data["Ch"] == 0]["ToT_ns"]
        data1_sample = sample_data[sample_data["Ch"] == 1]["ToT_ns"]

        sns.histplot(data0_sample, bins=num_bins, kde=True, color="blue", ax=axs[2, 0])
        sns.histplot(data1_sample, bins=num_bins, kde=True, color="red", ax=axs[2, 0])
        axs[2, 0].set_title("Approximate ToT distribution (Seaborn KDE)")
        axs[2, 0].set_xlabel("ToT in ns")

        # Hide the empty subplot at bottom right
        fig.delaxes(axs[2, 1])
        '''fig.delaxes(axs[1,0])''' #to cancel KDE approximation
        
        # Overall title
        fig.suptitle(r"ToT and $\Delta T$ in $ns$ for Channels 0 and 1 (all data, chunked)", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Leave room for title
        plt.show()

    except Exception as e:
        print(f"Error plotting histograms: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
        TDC_ToT_histograms(datafile)
    else:
        print("Couldn't filter data, no filtered input was given")
