from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.preprocessing.mqft import MQFT


def get_kwargs_code():
    """Used for generating directory names.

    Returns:
        dict: Key codes
    """
    return {
        "csvfiles": "CSV",
        "percentile_clipping": "P",
        "series_islogscale": "SL",  # .series log
        "series_isminmax": "SMM",
        "series_to_fft": "FFT",
        "series_to_mqft": "MQFT",  # .series log
        "freq_range": "F",
        "num_samples": "N",
    }


def get_output_dir(kwargs):
    """Generate output directory name.

    Args:
        kwargs (dict): Keyword arguments

    Returns:
        str: Directory name
    """
    codes = get_kwargs_code()
    dir_name = ""
    for key in kwargs.keys():
        if key in codes.keys():
            dir_name += f"{codes[key]}-{kwargs[key]}_"

    return dir_name[:-1]


def percentile_clip(data: np.ndarray, p: int = 90):
    """Clip/clamp the values below the `p` percentile to its value.

    Args:
        data (np.ndarray): _description_
        p (int, optional): Percentile value. Defaults to 90.

    Returns:
        _type_: _description_
    """
    data[data < np.percentile(data, p)] = np.percentile(data, p)
    return data


def freq_trans(data: np.ndarray, npoints: int, samples: int = 500):
    """Applies FFT to the data

    Args:
        data (np.ndarray): Data to apply FFT.
        npoints (int): Number of points.
        samples (int): Number of samples

    Returns:
        np.ndarray: Returns the FFT result
    """

    fft_result = scipy.fft.fft(data.values[:, samples - 1])
    pos_frequencies = 2 / npoints * np.abs(fft_result[0 : np.int_(npoints / 2)])
    return pos_frequencies


def normalize_numpy_data(data: np.ndarray):
    """Applies Logarithmic scaling and Minmax scaling on the data.

    Args:
        data (np.ndarray): Data to normalize

    Returns:
        np.ndarray: Returns the normalized data
    """
    # Apply normalization
    data = np.log(data)
    data_log_max = np.max(data)
    data_log_min = np.min(data)
    data_normed = np.abs(data - data_log_min) / np.abs(data_log_max - data_log_min)

    return data_normed


def generate_vectors(
    input_dir: Path,
    output_dir: Path,
    sampling_rate: int,
    start_time: int,
    end_time: int,
    num_samples: int = 1,
    **kwargs,
):
    """
    Generate time-frequency vectors from CSV files.

    Args:
        input_dir (Path): Input directory containing CSV files.
        output_dir (Path): Output directory for saving time-frequency vectors.
        sampling_rate (int): Specifies the sampling rate for time-frequency analysis.
        start_time (int): Specifies the start time for time-frequency analysis.
        end_time (int): Specifies the end time for time-frequency analysis.
        num_samples (int, optional): Number of samples to generate. Defaults to 1.

        **kwargs: Keyword arguments for processing CSV files.

    Raises:
        ValueError: If input or output directories are not directories.

    Kwargs:
        percentile_clipping: Specifies the percentile value for clipping.
        series_islogscale: Indicates whether logarithmic scaling should be applied to the series.
        series_isminmax: Indicates whether MinMax scaling should be applied to the series.
        series_to_fft: Indicates whether FFT should be applied to the series.
        series_to_mqft: Indicates whether MQFT should be applied to the series.
        freq_range: Specifies the frequency range for processing.
        num_samples: Specifies the number of samples to generate.
        chunksize: Specifies the chunk size for reading CSV files.
    """
    # Error checking
    # if not input_dir.is_dir() or not output_dir.is_dir():
    #     raise ValueError(
    #         f"Input and output directories must be directories. {input_dir} or {output_dir} are not directories."
    #     )

    # Prepare directory
    subdir_name = get_output_dir(kwargs)
    subdir_fullpath = Path(output_dir, subdir_name)
    subdir_fullpath.mkdir(parents=True, exist_ok=True)

    # Initialize scalers
    scaler = MinMaxScaler()
    csv_files = list(input_dir.glob("*.csv"))

    n_files = 0
    _tempdf = None

    for csv in tqdm(csv_files, desc="Generating vectors...", total=len(csv_files)):
        label = csv.stem  # FILENAME: Class1.csv

        if "chunksize" not in kwargs:
            _tempdf = pd.read_csv(csv, header=None)
        elif "chunksize" in kwargs:
            _tempdf = pd.DataFrame()
            chunks = pd.read_csv(csv, header=None, chunksize=kwargs["chunksize"])
            for chunk in chunks:
                _tempdf = pd.concat([_tempdf, chunk], ignore_index=True)

        if "series_isminmax" in kwargs and kwargs["series_isminmax"]:
            _tempdf = pd.DataFrame(
                scaler.fit_transform(_tempdf), columns=_tempdf.columns
            )

        # Get all samples if no num_samples specified
        _num_samples = len(_tempdf.columns) if num_samples < 0 else num_samples

        # Get the size of the array
        N = (end_time - start_time) * sampling_rate  # ARRAY SIZE: 725
        freq_range = kwargs.get("freq_range")

        for s in range(_num_samples):
            # Apply FFT to each sample
            if "series_to_fft" in kwargs and kwargs["series_to_fft"]:
                f1 = freq_trans(_tempdf, N, s)
                main_freq = np.array(f1).T
                ts = main_freq[:freq_range]

            # Apply MQFT to each sample
            elif "series_to_mqft" in kwargs and kwargs["series_to_mqft"]:
                mqft = MQFT(_tempdf.iloc[:, s - 1].values, fs=sampling_rate)
                mqft.run()
                ts = mqft.bft[:freq_range]

            else:
                ts = _tempdf.iloc[:freq_range, s]

            # Apply logarithmic scaling to each sample
            if "series_islogscale" in kwargs and kwargs["series_islogscale"]:
                y_log = np.log(ts)

                # Apply percentile clipping
                # Default: 90th percentile
                if (
                    "percentile_clipping" in kwargs
                    and kwargs["percentile_clipping"] < 100
                ):
                    y_log = percentile_clip(y_log, kwargs["percentile_clipping"])

                y_log_max = np.max(y_log)
                y_log_min = np.min(y_log)
                normalized = abs(y_log - y_log_min) / abs(y_log_max - y_log_min)
                ts = normalized

            else:
                # Apply percentage clipping
                if (
                    "percentile_clipping" in kwargs
                    and kwargs["percentile_clipping"] < 100
                ):
                    ts = percentile_clip(ts, kwargs["percentile_clipping"])

            # Save to directory
            save_dir = Path(subdir_fullpath, f"{label}_{s+1}.npy")
            np.save(file=save_dir, arr=ts)

        n_files += 1

    npy_files = list(subdir_fullpath.glob("*.npy"))
    print(f"{n_files} created npy files.")
    print(f"{len(npy_files)} npy files found in {subdir_fullpath}")