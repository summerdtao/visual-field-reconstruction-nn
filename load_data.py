import os
import numpy as np

from src.post_processing.data_processing.EEGProcessing import EEGProcessing
from src.settings import Settings
from src.pre_processing.split_source_image import splice_img
from src.post_processing.reconstruct_bitmappings import read_data, eeg_reconstruction

STIMULUS_DIR = Settings.STIMULUS_DATA_DIR
STIMULUS_FREQ = Settings.STIMULUS_FREQ
DATA_DIR = STIMULUS_DIR + 'eeg/'
TIMESTAMP_DIR = STIMULUS_DIR + 'index_timestamp/'
MASK_DIR = STIMULUS_DIR + 'flash_timestamp/'
SAMPLING_FREQ = Settings.MUSE_SAMPLING_FREQ
CONVOLUTION_WINDOW = Settings.CONVOLUTION_WINDOW
RANGE = [i for i in range(Settings.START, Settings.END + 1, Settings.STEP)]
IMG_NAME = Settings.IMG_STIMULUS.split('/')[-1].split('.')[0]


def process_experiment(process_func, data, timestamps, harmonics):
    bit_mapping = eeg_reconstruction(
        data,
        timestamps,
        harmonics,
        process_func=process_func
    )
    bit_mapping *= 127/np.average(bit_mapping)
    return bit_mapping


def extend_time_intervals(process_func):
    def amp_wrapper(data, start_time, end_time, harmonics=3):
        # Find the timestamp that match for the pixel we are looking for.
        masked_array = np.ma.masked_inside(data[0], start_time, end_time)
        # With the timestamps masked, get the eeg data for the wanted pixel.
        eeg_data = data[1][masked_array.mask]

        # Expand the time window until we have at least enough data to convolve on.
        if len(eeg_data) >= CONVOLUTION_WINDOW:
            return process_func(
                data,
                start_time,
                end_time,
                harmonics=harmonics
            )[0]
        else:
            return amp_wrapper(
                data,
                start_time - 0.05,
                end_time + 0.05,
                harmonics
            )

    return amp_wrapper


def get_inputs_with_labels():

    x, y = [], []
    data = read_data(DATA_DIR, 0)
    timestamps = read_data(TIMESTAMP_DIR, 0)

    amp = EEGProcessing(sampling_freq=SAMPLING_FREQ, stimulus_freq=STIMULUS_FREQ, total_time=160)

    for i in range(1, 4):
        bitmapping = process_experiment(
            extend_time_intervals(amp.fft_amplitude),
            data, timestamps, harmonics=i)
        x.append(np.asarray(bitmapping).reshape(RANGE[-1]) - RANGE[0])
    for i in range(1, 4):
        bitmapping = process_experiment(
            amp.integrated_phase_coherent_locked_in,
            data, timestamps, harmonics=i)
        x.append(np.asarray(bitmapping).reshape(RANGE[-1]) - RANGE[0])
    for i in range(1, 4):
        bitmapping = process_experiment(
            amp.dual_phase_lock_in,
            data, timestamps, harmonics=i)
        x.append(np.asarray(bitmapping).reshape(RANGE[-1]) - RANGE[0])

    eeg_averages = []
    for file_index in range(len(data)):
        experiment_data = data[file_index]
        experiment_timestamp = timestamps[file_index]

        experiment_timestamp = np.transpose(experiment_timestamp)
        experiment_data = np.transpose(experiment_data)
        subject_data_test = np.stack(
            (experiment_data[0], experiment_data[5]),
            axis=0
        )

        m = experiment_timestamp.shape
        for time_index in range(m[1] - 1):
            start_time = experiment_timestamp[0][time_index]
            end_time = experiment_timestamp[0][time_index + 1]
            end_time_offset = 0.2

            # Get the row eeg data for the wanted pixel using timestamps.
            masked_array = np.ma.masked_inside(subject_data_test[0], start_time, end_time - end_time_offset)
            eeg_data = subject_data_test[1][masked_array.mask]

            # Calculate averages
            eeg_averages.append(np.average(eeg_data))
        print("DONE {0}/{1} FILES".format(file_index + 1, len(data)))
    x.append(eeg_averages)

    x = np.asarray(x).transpose()
    y = splice_img(up=True, down=True, save_windows=False)
    y = y.reshape((y.shape[0] * y.shape[1]))[RANGE[0] : RANGE[-1]]

    return x, y


if __name__ == '__main__':
    x, y = get_inputs_with_labels()

    filename = f"ml/datasets/{Settings.NAME}_{Settings.STIMULUS_FREQ}_" \
               f"{IMG_NAME}_{Settings.TRIAL_NUM}_{Settings.START}_{Settings.END}.csv"
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    f = open(filename, 'w')
    for i in range(len(x)):
        for j in range(len(x[i])):
            f.write(str(x[i][j]) + ',')
        f.write(str(y[i]))
        f.write('\n')
    f.close()
