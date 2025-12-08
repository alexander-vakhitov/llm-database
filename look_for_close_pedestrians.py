import pandas
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv_path", required=True)
    parser.add_argument("-o", "--output", required=False)
    args = parser.parse_args()

    max_dist = 5

    path_to_file = args.csv_path
    df = pandas.read_csv(path_to_file, dtype={"aq_ts [ns]": np.int64})

    b_x = df[" bx"]
    b_y = df[" by"]
    distance_to_pedestrian = np.sqrt(b_x * b_x + b_y * b_y)
    timestamps = df[" aq_ts"]

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

    close_mask = dists_sorted < max_dist

    plt.scatter(timestamps_sorted[close_mask], dists_sorted[close_mask])

    # now, think about the first appearances of people. at what distance does it happen?
    intervals = timestamps_sorted[1:] - timestamps_sorted[:-1]
    large_interval_mask = intervals > 1e9  # intervals larger than 1 s
    distances_after_interval = dists_sorted[1:][large_interval_mask]

    plt.figure()
    plt.hist(distances_after_interval)

    timestamps_after_large_interval = timestamps_sorted[1:][large_interval_mask]

    thr = 2  # meters
    distance_less_thr_mask = distances_after_interval < thr

    print(
        f"{np.sum(distance_less_thr_mask)} times first appearance is closer than {thr} m"
    )
    plt.show()

    if args.output is not None:

        timestamps_close_after_interval = timestamps_after_large_interval[
            distance_less_thr_mask
        ]

        with open(args.output, "w") as file_out:
            for ts in timestamps_close_after_interval:
                file_out.write(str(ts) + "\n")
