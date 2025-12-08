import argparse
import numpy as np
from pathlib import Path
import pandas
import cv2


def read_event_timestamps(timestamps_path):
    event_timestamps = []
    with open(timestamps_path, "r") as file_input:

        for line in file_input:
            middle_timestamp = int(line)
            event_timestamps.append(middle_timestamp)

    return event_timestamps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timestamps", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-o", "--output-folder", required=True)
    args = parser.parse_args()

    cam0_csv_path = Path(args.dataset) / "data.csv"
    df = pandas.read_csv(cam0_csv_path)
    ts2file = {}
    for row in df.values:
        ts2file[int(row[0])] = Path(args.dataset) / "data" / row[1]

    all_timestamps = np.array(sorted(ts2file.keys()))

    interval = 5  # in seconds, before/after the event

    event_timestamps = read_event_timestamps(args.timestamps)

    for middle_timestamp in event_timestamps:
        start_timestamp = middle_timestamp - int(interval * 1e9)
        end_timestamp = middle_timestamp + int(interval * 1e9)
        timestamp_mask = (all_timestamps >= start_timestamp) & (
            all_timestamps <= end_timestamp
        )

        out = None
        for ts in all_timestamps[timestamp_mask]:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            img = cv2.imread(ts2file[ts], 0)
            if out is None:
                frame_height, frame_width = img.shape
                out = cv2.VideoWriter(
                    Path(args.output_folder) / f"{middle_timestamp}.mp4",
                    fourcc,
                    15.0,
                    (frame_width, frame_height),
                )
            out.write(img)

        if out is not None:
            out.release()
