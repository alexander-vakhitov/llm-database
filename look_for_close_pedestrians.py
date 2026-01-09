import pandas
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def get_velocities_for_timestamps(timestamps, path_to_trajectory_folder):
    path_to_smooth_trajectory = path_to_trajectory_folder / "smooth_trajectory_0.csv"
    df = pandas.read_csv(
        path_to_smooth_trajectory,
        dtype={"aq_ts [ns]": np.int64, "#timestamp [ns]": np.int64},
    )
    smooth_timestamps = np.array(df["#timestamp [ns]"])
    smooth_x = np.array(df[" p_RS_R_x [m]"])
    smooth_y = np.array(df[" p_RS_R_y [m]"])
    dt = (smooth_timestamps[1:] - smooth_timestamps[:-1]) / 1e9
    dx = smooth_x[1:] - smooth_x[:-1]
    vel_x = dx / dt
    vel_y = (smooth_y[1:] - smooth_y[:-1]) / dt
    vel_timestamps = smooth_timestamps[1:]
    velocities = -1 * np.ones((len(timestamps)))
    for i, ts in enumerate(timestamps):
        min_ind = np.argmin(np.abs(vel_timestamps - ts))
        vx = vel_x[min_ind]
        vy = vel_y[min_ind]
        velocities[i] = np.sqrt(vx * vx + vy * vy)
    return velocities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trajectory_path", required=True)
    parser.add_argument("-o", "--output", required=False)
    args = parser.parse_args()

    max_dist = 5
    vel_thr = 0.5

    path_to_trajectory_folder = Path(args.trajectory_path)
    path_to_file = path_to_trajectory_folder / "perception_bboxes.txt"
    df = pandas.read_csv(path_to_file, dtype={"aq_ts [ns]": np.int64})

    b_x = df[" bx"]
    b_y = df[" by"]
    distance_to_pedestrian = np.sqrt(b_x * b_x + b_y * b_y)
    timestamps = df["#hw_ts"]

    dT_min = (timestamps[len(timestamps) - 1] - timestamps[0]) / 1e9 / 60
    print(f"Dataset lasts {dT_min} minutes")

    ts_to_distance = {}
    for i in range(0, len(timestamps)):
        if timestamps[i] in ts_to_distance:
            ts_to_distance[timestamps[i]] = np.min(
                [ts_to_distance[timestamps[i]], distance_to_pedestrian[i]]
            )
        else:
            ts_to_distance[timestamps[i]] = distance_to_pedestrian[i]

    timestamps_sorted = np.array(sorted(list(ts_to_distance)))
    dists_sorted = np.array([ts_to_distance[ts] for ts in timestamps_sorted])

    # now, think about the first appearances of people. at what distance does it happen?
    intervals = timestamps_sorted[1:] - timestamps_sorted[:-1]
    large_interval_mask = intervals > 1e9  # intervals larger than 1 s
    distances_after_interval = dists_sorted[1:][large_interval_mask]

    plt.figure()
    plt.hist(distances_after_interval)

    timestamps_after_large_interval = timestamps_sorted[1:][large_interval_mask]

    velocities = get_velocities_for_timestamps(
        timestamps_after_large_interval, path_to_trajectory_folder
    )

    plt.figure()
    plt.hist(velocities)

    distance_less_thr_mask = distances_after_interval < max_dist
    large_vel = velocities > vel_thr

    distance_less_thr_mask = distance_less_thr_mask & large_vel

    print(
        f"{np.sum(distance_less_thr_mask)} times first appearance is closer than {max_dist} m"
    )
    plt.show()

    if args.output is not None:

        timestamps_close_after_interval = timestamps_after_large_interval[
            distance_less_thr_mask
        ]

        with open(args.output, "w") as file_out:
            for ts in timestamps_close_after_interval:
                file_out.write(str(ts) + "\n")
