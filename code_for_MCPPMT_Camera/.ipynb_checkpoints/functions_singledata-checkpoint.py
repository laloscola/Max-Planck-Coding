import pandas as pd
import numpy as np
import scipy as sp
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import csv
import pyarrow as pa 
import RS_RTxReadBin as rb  
from RS_RTxReadBin.RTxReadBin import RTxReadBin  
import os 
from scipy.optimize import curve_fit


# --------------------- 0. Converting csv to formatted csv and parquet -------------------

### converter for 3 channels oscilloscope.csv -> cleaned.csv //have to add correct time-step resolution
RESOLUTION = 2e-11
BUFFER_SIZE = 50_000  # number of rows to buffer before writing

def convert3channels_csv_to_csv_and_parquet(in_path, out_csv_path, out_parquet_path):
    buffer = []
    event_id = -1
    t0 = None
    idx = 0

    parquet_writer = None

    with open(in_path) as fin, open(out_csv_path, "w") as fout:
        # write CSV header
        fout.write("event_id,timestamp,Ch_1,Ch_2,Ch_3\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split(";")

            # new event
            if len(parts) == 1:
                event_id += 1
                t0 = float(parts[0])
                idx = 0
                continue

            # data row
            if len(parts) == 3:
                timestamp = t0 + idx * RESOLUTION
                idx += 1
                x, y, z = map(float, parts)

                # add to buffer
                buffer.append([event_id, timestamp, x, y, z])

                # flush buffer if full
                if len(buffer) >= BUFFER_SIZE:
                    # write CSV
                    for row in buffer:
                        fout.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

                    # write Parquet
                    df = pd.DataFrame(buffer, columns=["event_id", "timestamp", "Ch_1", "Ch_2", "Ch_3"])
                    table = pa.Table.from_pandas(df)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(out_parquet_path, table.schema, compression='snappy')
                    parquet_writer.write_table(table)

                    buffer = []

                continue

            raise ValueError(f"Unexpected format: {line}")

        # flush remaining buffer
        if buffer:
            for row in buffer:
                fout.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
            df = pd.DataFrame(buffer, columns=["event_id", "timestamp", "Ch_1", "Ch_2", "Ch_3"])
            table = pa.Table.from_pandas(df)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(out_parquet_path, table.schema, compression='snappy')
            parquet_writer.write_table(table)

        if parquet_writer:
            parquet_writer.close()

### converter for 2 channels oscilloscope.csv -> cleaned.csv //have to add correct time-step resolution
RESOLUTION = 2e-11
BUFFER_SIZE = 50_000  # number of rows to buffer before writing

def convert2channels_csv_to_csv_and_parquet(in_path, out_csv_path, out_parquet_path):
    buffer = []
    event_id = -1
    t0 = None
    idx = 0

    parquet_writer = None

    with open(in_path) as fin, open(out_csv_path, "w") as fout:
        # write CSV header
        fout.write("event_id,timestamp,Ch_1,Ch_2\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split(";")

            # new event
            if len(parts) == 1:
                event_id += 1
                t0 = float(parts[0])
                idx = 0
                continue

            # data row
            if len(parts) == 2:
                timestamp = t0 + idx * RESOLUTION
                idx += 1
                x, y = map(float, parts)

                # add to buffer
                buffer.append([event_id, timestamp, x, y])

                # flush buffer if full
                if len(buffer) >= BUFFER_SIZE:
                    # write CSV
                    for row in buffer:
                        fout.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

                    # write Parquet
                    df = pd.DataFrame(buffer, columns=["event_id", "timestamp", "Ch_1", "Ch_2"])
                    table = pa.Table.from_pandas(df)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(out_parquet_path, table.schema, compression='snappy')
                    parquet_writer.write_table(table)

                    buffer = []

                continue

            raise ValueError(f"Unexpected format: {line}")

        # flush remaining buffer
        if buffer:
            for row in buffer:
                fout.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
            df = pd.DataFrame(buffer, columns=["event_id", "timestamp", "Ch_1", "Ch_2"])
            table = pa.Table.from_pandas(df)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(out_parquet_path, table.schema, compression='snappy')
            parquet_writer.write_table(table)

        if parquet_writer:
            parquet_writer.close()

# -------------------------- 1. ratio analysis after conversion -------------------------

def load_df(data): # load data no matter if already df or file
    # Case 1: filename 
    if isinstance(data, str):
        df = pd.read_csv(data)   
    
    # Case 2: already a DataFrame 
    elif isinstance(data, pd.DataFrame):
        df = data
    
    else:
        raise TypeError("Input must be a filename or a pandas DataFrame.")
    return df    
    
def plot_all_channels(df, target_event = 0):
    # filter for target event
    event= df[df['event_id'] == target_event] 

    # plot
    plt.figure(figsize=(6,2))
    plt.plot(event["timestamp"], event["Ch_1"], label="Laser Trigger")
    plt.plot(event["timestamp"], event["Ch_2"], label="MCP PMT Camera")
    # plt.plot(event["timestamp"], event["Ch_3"], label="HPD")
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Signal")
    plt.title(f"Event {target_event}")
    plt.legend()
    plt.show()

def plot_only_camera(df, target_event = 0):
    # filter for target event
    event= df[df['event_id'] == target_event] 

    # plot
    plt.figure(figsize=(6,2))
    plt.plot(event["timestamp"], event["Ch_2"], label="MCP PMT Camera")
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Signal")
    plt.title(f"Event {target_event}")
    plt.legend()
    plt.show()

def interesting_value_filter(df, min_thresh = -0.01):  # filters interesting events according to thresh
    # one grouped min over the whole DataFrame
    mins = df.groupby('event_id')['Ch_2'].min()
    # select only those event_ids whose min < threshold
    interesting_ids = mins[mins < min_thresh].index.to_list()
    return interesting_ids

def main_ratio_and_plots(data, min_thresh = -0.01, show_all_channels=False, show_only_camera=True):
    '''Usage:
        1. Input of the cleaned file after convert_csv_to_csv_and_parquet
        2. Define acceptable threshold for the minima one wants to accept
       Purpose:
        - takes an oscilloscope dataset and returns the ratio of 
    '''
    df = load_df(data)  # returns df if either pandas dataframe or directly from file
    interesting_ids = interesting_value_filter(df, min_thresh)
    
    ratio = len(interesting_ids)/(max(df['event_id'])+1-len(interesting_ids))
    percentage = ratio * 100
    print(f'There are: {len(interesting_ids)} interesting events, that match the used threshold')
    # print(f'These events are numbered: {interesting_ids}')
    print(f'The percentage of interesting events to empty events is: {percentage:.3f}%')

    if show_all_channels:
        for i in interesting_ids:
            plot_all_channels(df, target_event=i)

    if show_only_camera:
        for i in interesting_ids:
            plot_only_camera(df, target_event=i)

    return percentage


# ----------------------------- 2. Calculating Jitter ------------------------------------

def peak_time_from_trace(t, y, height=None, prominence=None):  # calculates peak timing using find_peaks from scipy
    peaks, props = find_peaks(y, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None  # or raise
    # choose the highest / most prominent peak
    i = peaks[np.argmin(props["peak_heights"])]
    return t[i]

def cf_neg_pulse_time(t, y, frac=0.9):  # calculates the first element idx that is above the thresh 
    """
    Constant-fraction time pickoff. 
    Assumes a negative pulse on top of baseline-subtracted data.
    """
    A = y.min()
    if A >= 0:
        return None
    thr = frac * A # calculates max Amplitude A and determines the thresh based on A 

    idx = np.where(y <= thr)[0] # selects first index where y is above thresh -> timing based on first flank
    if len(idx) == 0:
        return None
        
    i1 = idx[0] # renaming
    if i1 == 0:
        return t[0]
        
    i0 = i1 - 1 # shifting by one index for linear interepolation

    t0, t1 = t[i0], t[i1]
    y0, y1 = y[i0], y[i1]
    if y1 == y0:
        return t1  # fallback
    # linear interpolation for values that satisfy threshold
    return t0 + (thr - y0) * (t1 - t0) / (y1 - y0)

def cf_pos_pulse_time(t, y, frac=0.7): # time pickoff via constant-fraction for one event
    """
    Constant-fraction time pickoff.
    Assumes a positive pulse on top of baseline-subtracted data.
    """
    A = y.max()
    if A <= 0:
        return None
    thr = frac * A

    idx = np.where(y >= thr)[0]
    if len(idx) == 0:
        return None
    i1 = idx[0]
    if i1 == 0:
        return t[0]
    i0 = i1 - 1

    t0, t1 = t[i0], t[i1]
    y0, y1 = y[i0], y[i1]
    if y1 == y0:
        return t1  # fallback
    # linear interpolation
    return t0 + (thr - y0) * (t1 - t0) / (y1 - y0)

def subtract_baseline(t, y, t_pre_max=None):
    """
    Subtract baseline estimated from pre-signal region.
    If t_pre_max is None, use first 10% of the time window.
    """
    if t_pre_max is None:
        t_pre_max = t.min() + 0.1 * (t.max() - t.min())
    mask_pre = t < t_pre_max
    if not np.any(mask_pre):
        baseline = np.median(y)
    else:
        baseline = np.median(y[mask_pre])
    return y - baseline, baseline

def event_time_diff(event_df, ch_ref="Ch_1", ch_test="Ch_2",
                    frac=0.9, t_pre_max=None):
    """
    Compute t_test - t_ref for a single event DataFrame.
    Returns None if timing cannot be determined.
    """
    t = event_df["timestamp"].to_numpy()

    y_ref = event_df[ch_ref].to_numpy()
    y_ref_bs, _ = subtract_baseline(t, y_ref, t_pre_max)
    t_ref = cf_pos_pulse_time(t, y_ref_bs, frac=frac)

    y_test = event_df[ch_test].to_numpy()
    y_test_bs, _ = subtract_baseline(t, y_test, t_pre_max)
    t_test = cf_neg_pulse_time(t, y_test_bs, frac=frac)

    if t_ref is None or t_test is None:
        return None
    return t_test - t_ref

def compute_jitter(df, ch_ref="Ch_1", ch_test="Ch_2",
                   frac=0.5, t_pre_max=None):
    """
    Loop over all event_id in df, compute time differences (test - ref),
    and return the list of deltas plus mean and std (jitter).
    """
    deltas = []
    for eid in df["event_id"].unique():
        event = df[df["event_id"] == eid].sort_values("timestamp")
        dt = event_time_diff(event, ch_ref=ch_ref, ch_test=ch_test,
                             frac=frac, t_pre_max=t_pre_max)
        if dt is not None:
            deltas.append(dt)

    deltas = np.array(deltas)
    if len(deltas) == 0:
        return deltas, None, None

    mean_offset = deltas.mean()
    jitter = deltas.std(ddof=1)  # jitter
    return deltas, mean_offset, jitter

def plot_jitter_histogram(deltas, bins=200, Run_Number = None, Counts=None, percentage = None):  # plot of the jitter distribution
    plt.figure(figsize=(5, 3))
    plt.hist(deltas, bins=bins, alpha=0.8, edgecolor="k")  # example: ps
    plt.xlabel("Time difference (s)")
    plt.ylabel("Counts")
    plt.title(f"Jitter distribution of 1 Mio Acquisitions")
    plt.tight_layout()
    # plt.savefig(f"JitterHist_Run{Run_Number}_Counts{Counts}.png", dpi=300)
    plt.show()
    
def plot_jitter_vs_event(deltas, Run_Number = None, Counts=None, percentage = None):  # if one wishes to find the deviating events
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(len(deltas)), deltas , ".", ms=3)
    plt.xlabel("Event index")
    plt.ylabel("Time difference (s)")
    plt.title(f"Jitter vs event of Run {Run_Number} with {Counts} acquisitions at ratio {percentage}%")
    plt.tight_layout()
    # plt.savefig(f"JitterVsEventHist_Run{Run_Number}_Counts{Counts}.png", dpi=300)
    plt.show()

def gauss(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C

def fit_gaussian_to_jitter(deltas, bins=50, range=None):
    """
    Fit a Gaussian to the jitter histogram (in seconds).
    Returns:
        popt  : [A, mu, sigma, C]
        pcov  : covariance matrix
        bin_centers, counts : histogram data used for fit
    """
    counts, bin_edges = np.histogram(deltas, bins=bins, range=range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if counts.max() == 0:
        return None, None, bin_centers, counts

    # initial guesses from histogram maximum
    i_max = np.argmax(counts)
    mu0 = bin_centers[i_max]
    sigma0 = np.std(deltas)
    if sigma0 == 0:
        sigma0 = (bin_centers[-1] - bin_centers[0]) / 10.0
    A0 = counts.max()
    C0 = counts.min()

    p0 = [A0, mu0, sigma0, C0]

    try:
        popt, pcov = curve_fit(gauss, bin_centers, counts, p0=p0)
    except RuntimeError:
        return None, None, bin_centers, counts

    return popt, pcov, bin_centers, counts

def plot_jitter_gaussian_two_views(deltas,
                                   bins=50,
                                   unit_scale=1e9,
                                   unit_label="ns",
                                   zoom_factor=2.0, 
                                   Run_Number = None , Counts=None , percentage = None, save_full_fig=False, save_zoomed_fig=False):
    """
    Plot jitter histogram + Gaussian fit in two views:
      1) Full x-range
      2) Zoomed around the peak (± zoom_factor * FWHM/2)
    deltas are in seconds; unit_scale converts to desired units for plotting.
    """
    # --- Fit Gaussian on histogram in seconds ---
    popt, pcov, bin_centers_s, counts = fit_gaussian_to_jitter(deltas, bins=bins)
    if popt is None:
        print("Gaussian fit to jitter histogram failed.")
        return None, None, None

    A, mu_s, sigma_s, C = popt
    sigma_s = np.abs(sigma_s)
    fwhm_s = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_s  # Gaussian FWHM [web:202]

    # Convert to user units
    deltas_u = deltas * unit_scale
    bin_centers_u = bin_centers_s * unit_scale
    mu_u = mu_s * unit_scale
    sigma_u = sigma_s * unit_scale
    fwhm_u = fwhm_s * unit_scale

    # x-grid for fitted curve in user units
    x_fit_u = np.linspace(bin_centers_u.min(), bin_centers_u.max(), 5000)
    x_fit_s = x_fit_u / unit_scale
    y_fit = gauss(x_fit_s, *popt)

    left_u = mu_u - 0.5 * fwhm_u
    right_u = mu_u + 0.5 * fwhm_u

    # ---------- 1) Full view ----------
    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4))
    plt.hist(deltas_u, bins=bins, alpha=0.7, color='C1', label="Data")
    plt.plot(x_fit_u, y_fit, "r-", label="Gaussian fit")

    plt.axvline(left_u, color="green", linestyle="--", label="FWHM")
    plt.axvline(right_u, color="green", linestyle="--")

    plt.xlabel(f"Time difference ({unit_label})")
    plt.ylabel("Counts")
    plt.title(f"Jitter distribution (full) for 1 Mio Acquisitions")

    plt.text(
        0.05, 0.95,
        f"μ ≈ {mu_u:.3f} {unit_label}\nσ ≈ {sigma_u:.3f} {unit_label}\nFWHM ≈ {fwhm_u:.3f} {unit_label}",
        transform=plt.gca().transAxes,
        va="top", ha="left", color="black"
    )

    plt.tight_layout()
    plt.legend()
    if save_full_fig:
        plt.savefig(f"JitterFitFullView_Run{Run_Number}_Counts{Counts}.png", dpi=300)
    plt.show()

    # ---------- 2) Zoomed view ----------
    half_width_u = 0.5 * fwhm_u * zoom_factor
    x_min = mu_u - half_width_u
    x_max = mu_u + half_width_u

    plt.figure(figsize=(6, 4))
    plt.hist(deltas_u, bins=bins, alpha=0.7, color='C1', edgecolor='k', linewidth=0, label="Data")
    plt.plot(x_fit_u, y_fit, "r-", label="Gaussian fit")

    plt.axvline(left_u, color="green", linestyle="--", label="FWHM")
    plt.axvline(right_u, color="green", linestyle="--")

    plt.xlim(x_min, x_max)
    plt.xlabel(f"Time difference ({unit_label})")
    plt.ylabel("Counts")
    plt.title(f"Jitter distribution (zoomed) for 1 Mio Acqusitions")

    plt.text(
        0.05, 0.95,
        f"μ ≈ {mu_u:.3f} {unit_label}\nσ ≈ {sigma_u:.3f} {unit_label}\nFWHM ≈ {fwhm_u:.3f} {unit_label}",
        transform=plt.gca().transAxes,
        va="top", ha="left", color="black"
    )

    plt.tight_layout()
    plt.legend()
    if save_zoomed_fig:
        plt.savefig(f"JitterFitZoomed_Run{Run_Number}_Counts{Counts}.png", dpi=300)
    plt.show()

    return mu_s, sigma_s, fwhm_s

def main_jitter_plot_and_fwhm(data, min_thresh=-0.01, bins_fit = 50, zoom_factor= 4, Run_Number = None, Counts=None, percentage = None, show_plots=True, save_fig=False):
    df = load_df(data)
    int_ids = interesting_value_filter(df, min_thresh=min_thresh)

    int_df = df[df['event_id'].isin(int_ids)]
    deltas, mean_offset, jitter = compute_jitter(int_df)
    if show_plots:
        plot_jitter_histogram(deltas, Run_Number=Run_Number, Counts=Counts, percentage=percentage)
        plot_jitter_vs_event(deltas, Run_Number=Run_Number, Counts=Counts, percentage=percentage)
        plot_jitter_gaussian_two_views(deltas, bins=bins_fit, zoom_factor=zoom_factor, Run_Number=Run_Number, Counts=Counts, percentage=percentage, save_fig=save_fig)

    return deltas, mean_offset, jitter 


# -------------------- 3. Integrating Pulse for Amplitude Spectrum --------------------------

def integrate_pulse(t, y, t_start, t_end):
    """
    Numerical integral of y(t) between t_start and t_end (trapezoidal rule).
    For negative pulses on baseline-subtracted data this area will be negative.
    """
    mask = (t >= t_start) & (t <= t_end)
    if not np.any(mask):
        return 0.0
    return np.trapezoid(y[mask], t[mask])  # V * s if y in V and t in s

def adaptive_window_negative(t, y, frac_level=0.1):
    """
    Find start/end of a negative pulse where it crosses a fraction of its depth.
    frac_level is fraction of |min|, e.g. 0.1 = 10% depth.
    """
    A = y.min()
    if A >= 0:
        return None, None
    thr = frac_level * A  # negative

    # indices where pulse is below this threshold
    idx = np.where(y <= thr)[0]
    if len(idx) == 0:
        return None, None

    i_start = idx[0]
    i_end = idx[-1]
    return t[i_start], t[i_end]

def event_charge_negative_adaptive(event_df, channel="Ch_2", cf_frac=0.5, level_frac=0.1, t_pre_max=None):
    """
    Baseline-subtract, locate negative pulse, then integrate over an
    amplitude-adaptive window (where |y| > level_frac * |min|).
    """
    t = event_df["timestamp"].to_numpy()
    y = event_df[channel].to_numpy()

    y_bs, baseline = subtract_baseline(t, y, t_pre_max=t_pre_max)

    # ensure a pulse exists
    if y_bs.min() >= 0:
        return None

    # use adaptive window from fraction-of-depth
    t_start, t_end = adaptive_window_negative(t, y_bs, frac_level=level_frac)
    if t_start is None:
        return None

    area = integrate_pulse(t, y_bs, t_start, t_end)
    return abs(area)  # V·time

def main_amplitude_spectrum(data, min_thresh=-0.01, bins = 100, channel = 'Ch_2', cf_frac = 0.5, level_frac = 0.1, show_plots=True, save_plots=False, Run_Number = None, Counts=None, percentage = None):
    df = load_df(data)
    int_ids = interesting_value_filter(df, min_thresh=min_thresh)
    int_df = df[df['event_id'].isin(int_ids)] # keep interesting df

    charges = []
    for eid in int_df["event_id"].unique():
        event = int_df[int_df["event_id"] == eid].sort_values("timestamp")
        q = event_charge_negative_adaptive(event, channel=channel, cf_frac = cf_frac, level_frac = level_frac)
        if q is not None:
            charges.append(q)
    
    if charges is not None: 
        print(f'The mean area is: {np.mean(charges)} Vs')
        if show_plots:  
            plt.figure(figsize=(5, 3))
            plt.hist(charges, bins=bins, alpha=0.8, edgecolor="k")
            plt.xlabel(f"Area under Pulse (Vs)")
            plt.ylabel("Counts")
            plt.title(f"Amplitude spectrum of Run {Run_Number} with {Counts} Aquisitions and Ratio {percentage}%")
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"AmplitudeSpectrum_Run{Run_Number}_Counts{Counts}.png", dpi=300)       
            plt.show()
        return charges

def main_main(data, bins_fit=1000, Run_Number=None , Counts=None):
    percentage = main_ratio_and_plots(data=data, min_thresh=-0.01, show_all_channels=False, show_only_camera=False)
    percentage = round(percentage, 3)
    
    main_jitter_plot_and_fwhm(data=data, bins_fit=bins_fit, Run_Number=Run_Number, Counts=Counts, percentage=percentage)
    main_amplitude_spectrum(data=data, Run_Number=Run_Number, Counts=Counts, percentage=percentage)
        
def main_amplitude_spectrum_zoom(data, min_thresh=-0.01, bins = 100, channel = 'Ch_2', cf_frac = 0.5, level_frac = 0.1, Run_Number = None, Counts=None, percentage = None):
    df = load_df(data)
    int_ids = interesting_value_filter(df, min_thresh=min_thresh)
    int_df = df[df['event_id'].isin(int_ids)] # keep interesting df

    charges = []
    for eid in int_df["event_id"].unique():
        event = int_df[int_df["event_id"] == eid].sort_values("timestamp")
        q = event_charge_negative_adaptive(event, channel=channel, cf_frac = cf_frac, level_frac = level_frac)
        if (q is not None) and (q <= 2e-10):
            charges.append(q)
    
    if charges is not None: 
        #print(f'The mean area is: {np.mean(charges)} Vs')
        
        plt.figure(figsize=(5, 3))
        plt.hist(charges, bins=bins, alpha=0.8, edgecolor="k")
        plt.xlabel(f"Area under Pulse (Vs)")
        plt.ylabel("Counts")
        plt.title(f"(zoomed) Amplitude spectrum of Run {Run_Number} with {Counts} Aquisitions and Ratio {percentage}%")
        plt.tight_layout()
        # plt.savefig(f"ZoomedAmplitudeSpectrum_Run{Run_Number}_Counts{Counts}.png", dpi=300)        
        plt.show()

 



# ------------------------------------- 4. binary format conversion 15th december -------------------------------------------
def converter_binary_to_df(filepath, show_info = True): 
    '''Should give pandas dataframe of the form: 
       event_id | time_axis | Ch_1_volt | Ch_2_volt |
    '''
    if os.path.exists(filepath): # checks existence of path
        root, ext = os.path.splitext(filepath)
        ext = ext.lower()
    
        if ext == '.bin': # checking file type
            y, x , S = RTxReadBin(filepath) # x is time axis, y is voltage, S is header information
            if show_info:
                for k in (list(S.keys())[:80]):
                    if k == 'Resolution': 
                        print(f'{k} -> {S[k]}')
                    if  k == 'RecordLength': 
                        print(f'{k} -> {S[k]}')
                    if k == 'Timestamps': 
                        print(f'Number of Acquisitions: {len(S[k])}\nTimestamps recorded: Yes')  # prints information about time resolution and recordlength           
        if ext == '.csv': 
            print('CSV file given: use meas_finalcode_camera.ipynb function convert#channels_csv_to_csv_and_parquet instead')
        if ext not in ('.bin', '.csv'): 
            print(f'Unknown filetype: {ext}')

        # now check for number of channels in y 
        
        n_samples, n_acq, n_ch = y.shape # how many samples per acquisition, acquisitions, channels
        if show_info:
            print(f'Number of channels given: {n_ch}')

        # broadcast x over acquisitions
        timestamps = np.tile(x[:, None], (1, n_acq))          # (n_samples, n_acq)
        event_ids = np.tile(np.arange(n_acq)[None, :], (n_samples, 1))  # (n_samples, n_acq)
        
        # flatten everything
        timestamps_flat = timestamps.ravel()   # length n_samples * n_acq
        event_ids_flat = event_ids.ravel()
        
        ch1_flat = y[:, :, 0].ravel()          # adapt indices to your channel mapping
        ch2_flat = y[:, :, 1].ravel()
        
        df = pd.DataFrame({
            "event_id": event_ids_flat,
            "timestamp": timestamps_flat,
            "Ch_1": ch1_flat,
            "Ch_2": ch2_flat,
        })

        return df     