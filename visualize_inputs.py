import pandas
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_to_file = "/media/alexander/slamcore_data/dhl/shift_trajectories/filtered_shift_trajectories/1/1_shift_20241125_0600.csv"
    df = pandas.read_csv(path_to_file, dtype={"acq_timestamp [ns]": np.int64})
    print(df.columns)
    print(np.unique(df["reference_frame_category"]))
    fid_world_mask = (
        df["reference_frame_category"] == "ReferenceFrameCategory.FiducialWorld"
    )
    print(f"Fiduial world poses: {np.sum(fid_world_mask)} out of {len(fid_world_mask)}")

    timestamps = df["acq_timestamp [ns]"].array
    time_intervals = timestamps[1:] - timestamps[:-1]
    print(time_intervals[10])
    print(f"Median time between poses {np.median(time_intervals) / 1e9}")

    time_subsampling_rate = 15

    timestamps = df["acq_timestamp [ns]"].array[fid_world_mask][::time_subsampling_rate]
    t_x = df["t_x [m]"].array[fid_world_mask][::time_subsampling_rate]
    t_y = df["t_y [m]"].array[fid_world_mask][::time_subsampling_rate]

    dx = t_x[1:] - t_x[:-1]
    dy = t_y[1:] - t_y[:-1]
    dt = timestamps[1:] - timestamps[:-1]
    continuous_diffs_mask = dt < 2 * 1e9
    velocities = np.sqrt(dx * dx + dy * dy)
    velocity_timestamps = timestamps[:-1][continuous_diffs_mask]
    velocities = velocities[continuous_diffs_mask]
    t_x = t_x[:-1][continuous_diffs_mask]
    t_y = t_y[:-1][continuous_diffs_mask]

    static_threshold = 0.05
    static_mask = velocities < static_threshold
    dynamic_mask = np.invert(static_mask)

    print(f"Static {np.sum(static_mask)} out of {len(static_mask)}")

    plt.figure()
    plt.scatter(t_x[dynamic_mask], t_y[dynamic_mask], color="g")
    plt.scatter(t_x[static_mask], t_y[static_mask], color="r")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    plt.figure()
    plt.plot(velocities)

    plt.show()
