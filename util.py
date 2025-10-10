import datetime
import numpy as np


def get_shift_time(timestamps, shift_type):
    earliest_time = datetime.datetime.fromtimestamp(np.min(timestamps) / 1e9)
    if shift_type == "day":
        shift_start = datetime.datetime(
            earliest_time.year, earliest_time.month, earliest_time.day, 6
        )
        shift_end = datetime.datetime(
            earliest_time.year, earliest_time.month, earliest_time.day, 18
        )
    else:
        shift_start = datetime.datetime(
            earliest_time.year, earliest_time.month, earliest_time.day, 18
        )
        shift_end = datetime.datetime(
            earliest_time.year, earliest_time.month, earliest_time.day, 6
        )
        if earliest_time.hour < 18:
            shift_start = shift_start - datetime.timedelta(days=1)
        else:
            shift_end = shift_end + datetime.timedelta(days=1)
    return shift_start, shift_end


def get_velocities(df, time_subsampling_rate):
    timestamps = df["acq_timestamp [ns]"].array[::time_subsampling_rate]
    t_x = df["t_x [m]"].array[::time_subsampling_rate]
    t_y = df["t_y [m]"].array[::time_subsampling_rate]
    data_mask = np.zeros((len(df["t_x [m]"].array)), dtype=bool)
    data_mask[::time_subsampling_rate] = True
    data_mask[0] = False
    dx = t_x[1:] - t_x[:-1]
    dy = t_y[1:] - t_y[:-1]
    dt = timestamps[1:] - timestamps[:-1]
    continuous_diffs_mask = dt < 2 * 1e9
    ref_frame_cat = df["reference_frame_category"].array[::time_subsampling_rate]
    ref_frame_ind = df["reference_frame_index"].array[::time_subsampling_rate]
    same_reference_frame_mask = (ref_frame_cat[1:] == ref_frame_cat[:-1]) * (
        ref_frame_ind[1:] == ref_frame_ind[:-1]
    )
    correct_velocity_mask = continuous_diffs_mask * same_reference_frame_mask
    velocities_abs = np.sqrt(dx * dx + dy * dy)
    headings_unnorm = np.stack(
        [dx[correct_velocity_mask], dy[correct_velocity_mask]], axis=1
    )
    velocities_abs = velocities_abs[correct_velocity_mask]
    nonzero_velocities_mask = velocities_abs > 0.01
    headings = np.zeros_like(headings_unnorm)
    headings[nonzero_velocities_mask] = headings_unnorm[
        nonzero_velocities_mask
    ] / np.linalg.norm(headings_unnorm[nonzero_velocities_mask], axis=1).reshape(-1, 1)
    velocity_timestamps = timestamps[:-1][correct_velocity_mask]

    data_mask[data_mask] = correct_velocity_mask

    return velocities_abs, headings, velocity_timestamps, data_mask


def get_activity_periods(min_time_interval_ns, shift_start, shift_end, timestamps):
    previous_timestamp = int(shift_start.timestamp()) * int(1e9)
    previous_recorded_timestamp = None
    time_periods = []
    for ts in timestamps:
        if ts - previous_timestamp > min_time_interval_ns:
            # let's make a record of an inactivity period > min_time_interval_ns
            if previous_recorded_timestamp is not None:
                time_periods.append(
                    (previous_recorded_timestamp, previous_timestamp, 1)
                )
            time_periods.append((previous_timestamp, ts, 0))
            previous_recorded_timestamp = ts
        previous_timestamp = ts

    last_timestamp = int(shift_end.timestamp()) * int(1e9)
    if previous_timestamp < last_timestamp:
        if previous_recorded_timestamp is not None:
            time_periods.append((previous_recorded_timestamp, previous_timestamp, 1))
        time_periods.append((previous_timestamp, last_timestamp, 0))

    return time_periods


def get_stopping_locations(df, data_mask, velocities, static_threshold_m):
    timestamps = df["acq_timestamp [ns]"].array[data_mask]
    static_mask = velocities < static_threshold_m
    t_x = df["t_x [m]"].array[data_mask]
    t_y = df["t_y [m]"].array[data_mask]
    stop_records = []
    stop_started = None
    stop_finished = None
    s_x = None
    s_y = None
    for i, ts in enumerate(timestamps):
        if static_mask[i]:
            if stop_started is None:
                stop_started = ts
                s_x = t_x[i]
                s_y = t_y[i]
            stop_finished = ts
        else:
            if stop_started is not None:
                stop_records.append((stop_started, stop_finished, s_x, s_y))
                stop_started = None
    if stop_started is not None:
        stop_records.append((stop_started, stop_finished, s_x, s_y))
    return stop_records
