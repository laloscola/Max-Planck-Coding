from itertools import zip_longest
import pyarrow.parquet as pq
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd 

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2.0 * sigma**2))

def fit_peak(centers, counts, around_mu=None, halfwidth=None):
    # pick default region around max if not given
    if around_mu is None:
        i_max = np.argmax(counts)
        around_mu = centers[i_max]
    if halfwidth is None:
        halfwidth = 1.0  # ns, adjust to your resolution
    mask = (centers >= around_mu - halfwidth) & (centers <= around_mu + halfwidth)
    x = centers[mask]
    y = counts[mask]
    if x.size < 5 or y.max() == 0:
        return None
    A0 = y.max()
    mu0 = x[y.argmax()]
    # crude sigma guess: halfwidth/2
    sig0 = max(halfwidth/2.0, (x[1]-x[0]) * 2.0)
    try:
        popt, pcov = curve_fit(gauss, x, y, p0=[A0, mu0, sig0], maxfev=10000)
        A, mu, sigma = popt
        fwhm = 2.35482004503 * abs(sigma)
        return dict(A=A, mu=mu, sigma=abs(sigma), fwhm=fwhm, x=x, y=y, cov=pcov)
    except Exception:
        return None

def bunching_histograms_parquet(path_sig, path_bkgrnd, batch_size=5_000_000, tot_range=(1, 5), delta_range=(0, 100), num_bins=135, columns=("ToT_ns","ToA_ns")):
    clock_start = time.perf_counter()  # start timer 

    print('Starting to accumulate histograms in batches...')

    pf_sig = pq.ParquetFile(path_sig)    # Parquet reader 
    pf_bkg = pq.ParquetFile(path_bkgrnd) # Parquet reader 

    it_sig = pf_sig.iter_batches(batch_size=batch_size, columns=list(columns))   # stream batches 
    it_bkg = pf_bkg.iter_batches(batch_size=batch_size, columns=list(columns))   # stream batches

    # bins and widths once
    bin_edges_tot   = np.linspace(tot_range[0], tot_range[1], num_bins + 1)     # shared ToT bins 
    bin_edges_delta = np.linspace(delta_range[0], delta_range[1], num_bins + 1) # shared delta bins 
    w_tot   = np.diff(bin_edges_tot)     # ToT bin widths
    w_delta = np.diff(bin_edges_delta)   # delta bin widths 

    # accumulators
    hist_sig_tot = np.zeros(num_bins, dtype=int)      # counts 
    hist_bkg_tot = np.zeros(num_bins, dtype=int)      # counts 
    hist_sig_delta = np.zeros(num_bins, dtype=int)    # counts 
    hist_bkggrnd_delta = np.zeros(num_bins, dtype=int)# counts 

    for i, (b1, b2) in enumerate(zip_longest(it_sig, it_bkg, fillvalue=None), start=1):
        print(f"Processing pair {i}...",
              f"file1={'present' if b1 is not None else 'None'}",
              f"file2={'present' if b2 is not None else 'None'}")  # progress 

        if b1 is not None:
            df_sig = b1.to_pandas()  # pandas chunk 
            if "ToT_ns" in df_sig:
                h, _ = np.histogram(df_sig["ToT_ns"].dropna(), bins=bin_edges_tot)  # ToT
                hist_sig_tot += h  # accumulate 
            if "ToA_ns" in df_sig:
                h, _ = np.histogram(df_sig["ToA_ns"].dropna(), bins=bin_edges_delta) # delta 
                hist_sig_delta += h  # accumulate

        if b2 is not None:
            df_bkg = b2.to_pandas()  # pandas chunk 
            if "ToT_ns" in df_bkg:
                h, _ = np.histogram(df_bkg["ToT_ns"].dropna(), bins=bin_edges_tot)  # ToT 
                hist_bkg_tot += h  # accumulate 
            if "ToA_ns" in df_bkg:
                h, _ = np.histogram(df_bkg["ToA_ns"].dropna(), bins=bin_edges_delta) # delta 
                hist_bkggrnd_delta += h  # accumulate

    # normalize to densities (delta only)
    n_sig_delta   = hist_sig_delta.sum()                      # total samples 
    n_bkg_delta   = hist_bkggrnd_delta.sum()                  # total samples 
    dens_sig      = (hist_sig_delta / (n_sig_delta * w_delta)) if n_sig_delta > 0 else np.zeros_like(hist_sig_delta, float)  # density 
    dens_bkg      = (hist_bkggrnd_delta / (n_bkg_delta * w_delta)) if n_bkg_delta > 0 else np.zeros_like(hist_bkggrnd_delta, float)  # density 
    resid_delta   = dens_sig - dens_bkg                       # residual density 
    centers_delta = 0.5 * (bin_edges_delta[:-1] + bin_edges_delta[1:])           # centers 
    centers_tot   = 0.5 * (bin_edges_tot[:-1] + bin_edges_tot[1:])               # centers 

    # plots
    plt.style.use('ggplot')  # style 
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # grid

    axs[0,0].bar(bin_edges_tot[:-1], hist_sig_tot, width=w_tot, align="edge", alpha=0.7, color="blue")  # ToT signal 
    axs[0,0].set_title("Signal ToT counts"); axs[0,0].set_xlabel("ToT ns"); axs[0,0].set_ylabel("Count")  # labels 
    axs[0,1].bar(bin_edges_tot[:-1], hist_bkg_tot, width=w_tot, align="edge", alpha=0.7, color="red")    # ToT background
    axs[0,1].set_title("Background ToT counts"); axs[0,1].set_xlabel("ToT ns")  # labels

    axs[1,0].bar(bin_edges_delta[:-1], hist_sig_delta, width=w_delta, align='edge', alpha=0.7, color='blue')  # delta signal 
    axs[1,0].set_title("Signal ToA counts"); axs[1,0].set_xlabel("ToA ns"); axs[1,0].set_ylabel("Count")  # labels 
    axs[1,1].bar(bin_edges_delta[:-1], hist_bkggrnd_delta, width=w_delta, align='edge', alpha=0.7, color='red')  # delta background 
    axs[1,1].set_title("Background ToA counts"); axs[1,1].set_xlabel("ToA ns")  # labels

    axs[2,0].bar(centers_delta, resid_delta, width=w_delta, align="center", alpha=0.8, color="purple")  # residual 
    axs[2,0].axhline(0, color='k', lw=1); axs[2,0].set_title("Delta residual (density)")  # baseline/title 
    axs[2,0].set_xlabel("ToA ns"); axs[2,0].set_ylabel("Density diff")  # labels 

    # Make axs[2,1] empty
    axs[2,1].axis('off')  # hide the last subplot 
    # Alternatively: fig.delaxes(axs[2,1])  # remove axis entirely 

    plt.tight_layout(); plt.show()  # layout and display 

    clock_time = time.perf_counter() - clock_start  # end timer
    print(f'Histograms have been successfully created in {clock_time:.2f}s')  # report 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect histogram data and calculate normalized bunching data")
    parser.add_argument("signal_file", help="Path to input Signal file")
    parser.add_argument("background_file", help="Path to input Background file")
    parser.add_argument("--batch_size", type=int, default=5_000_000, help="Rows per batch (default: 5,000,000)")
    parser.add_argument("--tot_min", type=float, default=1.0)
    parser.add_argument("--tot_max", type=float, default=5.0)
    parser.add_argument("--delta_min", type=float, default=0.0)
    parser.add_argument("--delta_max", type=float, default=100.0)
    parser.add_argument("--bins", type=int, default=135)
    args = parser.parse_args()

    bunching_histograms_parquet(
        args.signal_file,
        args.background_file,
        batch_size=args.batch_size,
        tot_range=(args.tot_min, args.tot_max),
        delta_range=(args.delta_min, args.delta_max),
        num_bins=args.bins,
    )