import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def photonbunching(datafile, chunksize=10_000_000, delay=8, window=10, num_bins=135):
    '''Remarks:
        - needs filtering beforehand by big_datafilter.py 
        - the delay of 8ns or 16ns depends on the setup and has to be plugged in manually -> maybe in terminal command line or maybe in the code itself
        - acceptance window for photonbunching has to be adapted afterwards depending on accepted errors
            -> maybe read up on errors of TDC, photomultipliers, general setup....
        - window parameter hasn't been used yet 
    Description: 
        - calculates difference of Channel 0 and Channel 1 deltaT_ns (deltaT_ns always has the common start reference point (depending on random window or coincidence circuit))
    Open questions:
        - check on which Channel the delay has been put
        - I could add an fitted curve that analyzes the bunching effect
        - Could decide to do the plots separately and just save the difference data in a file to further analyze 
        - Could also absorb big_bunching_plot.py in big_histograms.py 
    '''
    try:
        reader = pd.read_csv(datafile, sep='\t', skipinitialspace=True, chunksize=chunksize)
        hist=np.zeros(num_bins, dtype=int)
        
        for i, chunk in enumerate(reader):
            print(f'Processing chunk: {i+1}')
            
            mask_pairs = chunk[chunk['Ch'] == 0] # check if logic is correct to select the pairs that give the difference  
            difference = pd.DataFrame({
                "Tstamp_us": mask_pairs['Tstamp_us'],
                "TrgID": mask_pairs['TrgID'],
                "delta_diff": mask_pairs['deltaT_ns'] - mask_pairs['deltaT_ns'].shift(-1) - delay # check if + or - depending on which channel the delay has been put upon, for the test data we had no delay
            })
    
            # updating the histogram
            h, _ = np.histogram(difference['delta_diff'].dropna(), bins=num_bins)
            hist += h  # local histogram save 
        
        #plot of photonbunching 
        plt.style.use('ggplot')
        plt.rcParams['text.usetex']=False
        
        plt.hist(difference['delta_diff'], num_bins)
        plt.title('Photonbunching effect')
        plt.xlabel('Signal differences of Ch 0 and Ch 1 in ns')
        plt.ylabel('Frequency')
        plt.show()

    except Exception as e:
        print(f"Error plotting histograms: {e}")
        return None
            

if __name__ == "__main__":
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
        photonbunching(datafile)
    else:
        print("Couldn't filter data, no filtered input was given")