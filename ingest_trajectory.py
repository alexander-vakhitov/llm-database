import sqlite3
import os
import argparse
import pandas
import numpy as np
import datetime
import sys
from util import get_shift_time, get_activity_periods, get_velocities


def insert_activity(forklift_id, time_periods, cur, con):
    query_with_param = "INSERT INTO activity(forklift_id, start, end) VALUES(?, ?, ?)"
    for activity_record in time_periods:
        time_start, time_end, status = activity_record
        if status:
            time_start = datetime.datetime.fromtimestamp(int(time_start / 1e9))
            time_end = datetime.datetime.fromtimestamp(int(time_end / 1e9))
            cur.execute(query_with_param, (forklift_id, time_start, time_end))
    con.commit()


def insert_trajectory(
    forklift_id, velocity_timestamps, coordinates, headings, velocities_abs, cur, con
):
    query_with_param = "INSERT INTO trajectory(forklift_id, time, coordinate_horizontal, coordinate_vertical, heading_horizontal, heading_vertical, velocity_meters_per_second) VALUES (?, ?, ?, ?, ?, ?, ? )"
    for trajectory_record in zip(
        velocity_timestamps, coordinates, headings, velocities_abs
    ):
        time_ns, coordinate, heading, velocity_abs = trajectory_record
        time_moment = datetime.datetime.fromtimestamp(int(time_ns / 1e9))
        cur.execute(
            query_with_param,
            (
                forklift_id,
                time_moment,
                coordinate[0],
                coordinate[1],
                heading[0],
                heading[1],
                velocity_abs,
            ),
        )
    con.commit()


if __name__ == "__main__":
    time_subsampling_rate = 15
    min_time_interval_ns = 600 * 1e9
    is_insert_trajectory = True
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("forklift_id", type=int)
    parser.add_argument("shift_type", choices=["day", "night"])

    args = parser.parse_args()
    db_path = "aware_data.db"
    is_creating_tables = not os.path.exists(db_path)

    con = sqlite3.connect(
        db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )
    cur = con.cursor()
    if is_creating_tables:
        cur.execute(
            "CREATE TABLE activity(id int primary key, forklift_id int, start timestamp, end timestamp)"
        )
        if is_insert_trajectory:
            cur.execute(
                "CREATE TABLE trajectory(id int primary key, forklift_id int, time timestamp, coordinate_horizontal float, coordinate_vertical float, heading_horizontal float, heading_vertical float, velocity_meters_per_second float)"
            )
        con.commit()

    forklift_id = args.forklift_id

    if not os.path.exists(args.csv_path):
        sys.exit(0)
    df = pandas.read_csv(args.csv_path, dtype={"acq_timestamp [ns]": np.int64})

    # data processing params
    time_subsampling_rate = 15
    static_threshold = 0.05
    # data processing starts here
    velocities_abs, headings, velocity_timestamps, data_mask = get_velocities(
        df, time_subsampling_rate
    )

    static_mask = velocities_abs < static_threshold
    dynamic_mask = np.invert(static_mask)

    activity_timestamps = velocity_timestamps[dynamic_mask]

    # insert activities into DB
    shift_start, shift_end = get_shift_time(velocity_timestamps, args.shift_type)
    time_periods = get_activity_periods(
        min_time_interval_ns, shift_start, shift_end, activity_timestamps
    )

    insert_activity(forklift_id, time_periods, cur, con)

    if not is_insert_trajectory:
        exit()

    # insert trajectory (locations, headings, velocities) into DB
    fid_world_mask = (
        df["reference_frame_category"] == "ReferenceFrameCategory.FiducialWorld"
    )
    trajectory_mask = data_mask * fid_world_mask
    coordinates = np.stack(
        [df["t_x [m]"].array[trajectory_mask], df["t_y [m]"].array[trajectory_mask]],
        axis=1,
    )
    trajectory_mask_for_data_mask = trajectory_mask[data_mask]
    velocity_timestamps = velocity_timestamps[trajectory_mask_for_data_mask]
    headings = headings[trajectory_mask_for_data_mask]
    velocities_abs = velocities_abs[trajectory_mask_for_data_mask]

    print(
        f"Length of trajectory is {len(velocity_timestamps)} {len(coordinates)} {len(headings)} {len(velocities_abs)} "
    )
    insert_trajectory(
        forklift_id,
        velocity_timestamps,
        coordinates,
        headings,
        velocities_abs,
        cur,
        con,
    )

    # database filled
