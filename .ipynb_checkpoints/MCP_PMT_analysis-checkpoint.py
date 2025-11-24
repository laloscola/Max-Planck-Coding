import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd 
from scipy.optimize import curve_fit

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

def plot_FWHM_analysis(centers_delta, hist_MCP_delta, hist_HPD_delta):
    # 2) Zoomed delta peaks with fits
    fit_MCP = fit_peak(centers_delta, hist_MCP_delta, halfwidth=1.5)   # tune halfwidth
    fit_HPD = fit_peak(centers_delta, hist_HPD_delta, halfwidth=1.5)
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, counts, title, fit in [
        (axs2[0], hist_MCP_delta, "MCP delta peak", fit_MCP),
        (axs2[1], hist_HPD_delta, "HPD delta peak", fit_HPD),
    ]:
        ax.step(centers_delta, counts, where='mid', label='Histogram')
        if fit is not None:
            # smooth x for fitted curve
            xfit = np.linspace(fit['x'][0], fit['x'][-1], 400)
            yfit = gauss(xfit, fit['A'], fit['mu'], fit['sigma'])
            ax.plot(xfit, yfit, 'b-', lw=2, label='Gaussian fit')
            # FWHM lines
            halfmax = fit['A'] * np.exp(0) * 0.5
            # Solve for the two x at half max analytically for Gaussian:
            dx = np.sqrt(2*np.log(2)) * fit['sigma']
            xL, xR = fit['mu'] - dx, fit['mu'] + dx
            ax.hlines(halfmax, xL, xR, colors='k', linestyles='--', label='FWHM')
            ax.vlines([xL, xR], 0, halfmax, colors='k', linestyles=':')
            ax.text(fit['mu'], halfmax, f"FWHM = {fit['fwhm']:.3f} ns", ha='center', va='bottom')
            # zoom limits
            pad = 0.5 * fit['fwhm']
            ax.set_xlim(fit['mu'] - max(10*dx, pad), fit['mu'] + max(10*dx, pad))
            ax.set_ylim(0, max(counts[(centers_delta>=fit['mu']-2*dx)&(centers_delta<=fit['mu']+2*dx)].max()*1.2, yfit.max()*1.2))
        ax.set_title(title)
        ax.set_xlabel("ToA ns")
        ax.set_ylabel("Count")
        ax.legend()
    
    plt.tight_layout(); plt.show()
    

def accumulate_full_hists(file_path, tot_range=(0.001, 5), delta_range=(0, 20), batch_size = 500_000, num_bins=1000, columns = ('Tstamp_us', 'Ch', 'ToA_ns', 'ToT_ns')):
    
    pf = pq.ParquetFile(file_path)
    it = pf.iter_batches(batch_size = batch_size, columns=columns)
    
    # bins and widths once
    bin_edges_tot   = np.linspace(tot_range[0], tot_range[1], num_bins + 1)     # shared ToT bins 
    bin_edges_delta = np.linspace(delta_range[0], delta_range[1], num_bins + 1) # shared delta bins 
    w_tot   = np.diff(bin_edges_tot)     # ToT bin widths
    w_delta = np.diff(bin_edges_delta)   # delta bin widths 
    
    # accumulators
    hist_MCP_tot = np.zeros(num_bins, dtype=int)      # counts 
    hist_MCP_delta = np.zeros(num_bins, dtype=int)    # counts 
    hist_HPD_tot = np.zeros(num_bins, dtype=int)      # counts 
    hist_HPD_delta = np.zeros(num_bins, dtype=int)    # counts 
    hist_trig_tot = np.zeros(num_bins, dtype=int)      # counts 
    hist_trig_delta = np.zeros(num_bins, dtype=int)    # counts 
    
    it = pf.iter_batches(batch_size = 500_000, columns=None)
    
    print('Accumulating Histograms...')
    for i, batch in enumerate(it):
        # print(f'Processing batch {i}...')
        # print(batch)
        # mask = (pc.field('ToA_ns') >= delta_range[0]) & (pc.field('ToA_ns') <= delta_range[1])  # maybe add: & (pc.field('ToT_ns') != 0) & (pc.field('Entries') == 2)
    
        # splitting into Channels
        mask_MCP = pc.equal(batch["Ch"], 9) 
        pf_MCP = batch.filter(mask_MCP)
        mask_HPD = pc.equal(batch["Ch"], 8)
        pf_HPD = batch.filter(mask_HPD)
        mask_trig = pc.equal(batch['Ch'], 0)
        pf_trig = batch.filter(mask_trig)
        
        if pf_MCP.num_rows > 0: 
            df_MCP = pf_MCP.to_pandas()
            if 'ToT_ns' in df_MCP:   
                h, _ = np.histogram(df_MCP['ToT_ns'].dropna(), bins=bin_edges_tot)  # ToT
                hist_MCP_tot += h  # accumulate 
            if 'ToA_ns' in df_MCP:
                h, _ = np.histogram(df_MCP["ToA_ns"].dropna(), bins=bin_edges_delta) # delta 
                hist_MCP_delta += h  # accumulate
    
        if pf_HPD.num_rows > 0: 
            df_HPD = pf_HPD.to_pandas()
            if 'ToT_ns' in df_HPD:   
                h, _ = np.histogram(df_HPD['ToT_ns'].dropna(), bins=bin_edges_tot)  # ToT
                hist_HPD_tot += h  # accumulate 
            if 'ToA_ns' in df_HPD:
                h, _ = np.histogram(df_HPD["ToA_ns"].dropna(), bins=bin_edges_delta) # delta 
                hist_HPD_delta += h  # accumulate
    
        if pf_trig.num_rows > 0:
            df_trig = pf_trig.to_pandas()
            if 'ToT_ns' in df_trig:   
                h, _ = np.histogram(df_trig['ToT_ns'].dropna(), bins=bin_edges_tot)  # ToT
                hist_trig_tot += h  # accumulate 
            if 'ToA_ns' in df_trig:
                h, _ = np.histogram(df_trig["ToA_ns"].dropna(), bins=bin_edges_delta) # delta 
                hist_trig_delta += h  # accumulate
    
    
    centers_delta = 0.5 * (bin_edges_delta[:-1] + bin_edges_delta[1:])           # centers 
    centers_tot   = 0.5 * (bin_edges_tot[:-1] + bin_edges_tot[1:])               # centers 

    return (centers_tot, bin_edges_tot, centers_delta, bin_edges_delta,
            hist_MCP_tot, hist_HPD_tot, hist_trig_tot,
            hist_MCP_delta, hist_HPD_delta, hist_trig_delta)

    
def plot_full(centers_tot, bin_edges_tot, centers_delta, bin_edges_delta,
              hist_MCP_tot, hist_HPD_tot, hist_trig_tot,
              hist_MCP_delta, hist_HPD_delta, hist_trig_delta):
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: ToT histograms
    axs[0].hist(
        [centers_tot, centers_tot],
        bins=bin_edges_tot,
        weights=[hist_MCP_tot, hist_HPD_tot], # insert: hist_trig_tot here to display trigger data
        histtype="step",
        linewidth=1,
        label=["MCP", "HPD", "Trigger"],
    )
    axs[0].set_title("ToT counts")
    axs[0].set_xlabel("ToT ns")
    axs[0].set_ylabel("Count")
    axs[0].legend()
    
    # fig.delaxes(axs[0]) # deselect here to display ToT data
    
    # Right: ToA histograms 
    axs[1].hist(
        [centers_delta, centers_delta], 
        bins=bin_edges_delta,
        weights=[hist_MCP_delta, hist_HPD_delta],
        histtype="step",
        linewidth=1,
        label=["MCP", "HPD", "trigger"],
    )
    axs[1].set_title("ToA counts")
    axs[1].set_xlabel("ToA ns")
    axs[1].set_ylabel("Count")
    axs[1].legend()
    
    # comment: there are too few ToA datapoints for the Trigger (which makes sense) to display 
    # comment: in the ToT data we see a delta peak at ~ 3.2ns ,this is probably not physical (one sees this when comparing to trigger data), might just be what the TDC does when reaching its upper limit
    plt.tight_layout(); plt.show()

    



def parse_args():
    p = argparse.ArgumentParser(description="Accumulate histograms from Parquet and fit Gaussian to delta peak (FWHM)")
    p.add_argument("parquetfile", help="Path to Parquet file")
    p.add_argument("--bins", type=int, default=1000, help="Number of bins")
    p.add_argument("--tot_range", nargs=2, type=float, default=[0.001, 5.0], help="ToT range: min max (ns)")
    p.add_argument("--delta_range", nargs=2, type=float, default=[0.0, 20.0], help="Delta range: min max (ns)")
    p.add_argument("--batch", type=int, default=500_000, help="Batch size for iter_batches")
    return p.parse_args()
    
def main(): # sequentially runs the different functions
    # 1) collect arguments
    args = parse_args()

    # 2) Accumulate the histogram data and yield
    (centers_tot, bin_edges_tot, centers_delta, bin_edges_delta,
    hist_MCP_tot, hist_HPD_tot, hist_trig_tot,
    hist_MCP_delta, hist_HPD_delta, hist_trig_delta) = accumulate_full_hists(args.parquetfile, 
                                                                              num_bins = args.bins, 
                                                                              tot_range = tuple(args.tot_range),
                                                                              delta_range = tuple(args.delta_range),
                                                                              batch_size = args.batch)
    # 3) Plot full histograms
    plot_full(centers_tot, bin_edges_tot, centers_delta, 
              bin_edges_delta, hist_MCP_tot, hist_HPD_tot, hist_trig_tot, 
              hist_MCP_delta, hist_HPD_delta, hist_trig_delta)
    

    # 4) Run FWHM analysis with Gauss peak    
    plot_FWHM_analysis(centers_delta, hist_MCP_delta, hist_HPD_delta)


if __name__ == '__main__':
    main()