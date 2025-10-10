import pandas
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime, timedelta
import cv2
import slamcore

from matplotlib.backend_bases import MouseButton

from util import (
    get_shift_time,
    get_activity_periods,
    get_velocities,
    get_stopping_locations,
)


def get_meter_to_unit(session_path):
    session = slamcore.load_session_file(str(session_path))
    return session.floor_plan.unit_to_meters


def scatter_on_floorplan(x, y, s, c, meter_to_pixel, floorplan_height_pix):
    if floorplan_height_pix is None:
        return plt.scatter(x, y, color=c, s=s)
    else:
        return plt.scatter(
            meter_to_pixel * x,
            floorplan_height_pix - meter_to_pixel * y,
            color=c,
            s=s,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv_path", required=True)
    parser.add_argument("-s", "--shift_type", choices=["day", "night"], required=True)
    parser.add_argument("-m", "--map_session")
    parser.add_argument("-f", "--floorplan")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    meter_to_unit = get_meter_to_unit(args.map_session)

    path_to_file = args.csv_path
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

    # data processing params
    time_subsampling_rate = 15
    static_threshold_m = 0.05
    min_time_interval_ns = 600 * 1e9
    # data processing starts here
    velocities_abs, headings, velocity_timestamps, data_mask = get_velocities(
        df, time_subsampling_rate
    )

    static_mask = velocities_abs < static_threshold_m
    dynamic_mask = np.invert(static_mask)

    print(f"Static {np.sum(static_mask)} out of {len(static_mask)}")

    plt.figure()
    plt.plot(velocity_timestamps, velocities_abs)

    # forklift activity
    activity_timestamps = velocity_timestamps[dynamic_mask]
    date_time_format = "%Y-%m-%d %H:%M:%S"

    if len(activity_timestamps) > 0:
        earliest_timestamp = np.min(activity_timestamps)
        latest_timestamp = np.max(activity_timestamps)
        earliest_time = datetime.datetime.fromtimestamp(earliest_timestamp / 1e9)
        latest_time = datetime.datetime.fromtimestamp(latest_timestamp / 1e9)

        earliest_time_str = earliest_time.strftime(date_time_format)
        latest_time_str = latest_time.strftime(date_time_format)
        print(f"Active from {earliest_time_str} to {latest_time_str}")

    shift_start, shift_end = get_shift_time(timestamps[0], args.shift_type)
    shift_start_str = shift_start.strftime(date_time_format)
    shift_end_str = shift_end.strftime(date_time_format)
    print(f"Shift from {shift_start_str} to {shift_end_str}")

    time_periods = get_activity_periods(
        min_time_interval_ns, shift_start, shift_end, activity_timestamps
    )
    print(len(time_periods))
    eps = 1
    activity_status_time = []
    activity_status_value = []
    for activity_record in time_periods:
        time_start, time_end, status = activity_record
        activity_status_time.append(time_start + eps)
        activity_status_value.append(status)
        activity_status_time.append(time_end - eps)
        activity_status_value.append(status)

    plt.figure("Activity status")
    activity_status_time = np.array(activity_status_time).astype(int)
    # activity_status_time -= activity_status_time[0]
    activity_status_time = activity_status_time.astype(float) / 1e9
    plt.plot(activity_status_time, activity_status_value)

    # visualize stopping moments
    fid_world_data_mask = np.copy(data_mask)
    fid_world_data_mask = data_mask * fid_world_mask
    stopping_records = get_stopping_locations(
        df, fid_world_data_mask, velocities_abs, static_threshold_m
    )
    small_stop_thr = 60 * 1e9
    small_stops = []
    large_stops = []
    for stop_record in stopping_records:
        stop_started, stop_finished, s_x, s_y = stop_record
        if stop_finished - stop_started > small_stop_thr:
            large_stops.append([s_x, s_y])
        else:
            small_stops.append([s_x, s_y])

    print(f"Small stops {len(small_stops)} Large stops {len(large_stops)}")

    fig = plt.figure("Trajectories and stops")

    floorplan_height_pix = None
    if args.floorplan is not None:
        img_floorplan = cv2.imread(args.floorplan)
        plt.imshow(img_floorplan)
        meter_to_pixel = meter_to_unit
        floorplan_height_pix = img_floorplan.shape[0]
    else:
        meter_to_pixel = 1

    all_plots = []
    all_legends = []

    t_x = df["t_x [m]"][fid_world_data_mask]
    t_y = df["t_y [m]"][fid_world_data_mask]
    traj_plot = scatter_on_floorplan(
        t_x, t_y, 1, "b", meter_to_pixel, floorplan_height_pix
    )
    all_plots.append(traj_plot)
    all_legends.append("Trajectory")
    if len(large_stops) > 0:
        large_stops = np.array(large_stops)
        large_plot = scatter_on_floorplan(
            large_stops[:, 0],
            large_stops[:, 1],
            10,
            "r",
            meter_to_pixel,
            floorplan_height_pix,
        )
        all_plots.append(large_plot)
        all_legends.append("Large stops")
    if len(small_stops) > 0:
        small_stops = np.array(small_stops)
        small_plot = scatter_on_floorplan(
            small_stops[:, 0],
            small_stops[:, 1],
            5,
            "m",
            meter_to_pixel,
            floorplan_height_pix,
        )
        all_plots.append(small_plot)
        all_legends.append("Small stops")

    plt.legend(
        all_plots, all_legends, scatterpoints=1, loc="lower left", ncol=3, fontsize=8
    )

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    pt1_coords = None
    output_path = args.output
    area_list = []

    def on_click(event):
        global pt1_coords
        global fig
        global output_path
        if event.button is MouseButton.LEFT:
            print(
                f"data coords {event.xdata} {event.ydata},",
                f"pixel coords {event.x} {event.y}",
            )
            if pt1_coords is None:
                pt1_coords = [event.xdata, event.ydata]
            else:
                pt2_coords = [event.xdata, event.ydata]
                lx = np.min([pt2_coords[0], pt1_coords[0]])
                ly = np.min([pt2_coords[1], pt1_coords[1]])
                w = np.abs(pt2_coords[0] - pt1_coords[0])
                h = np.abs(pt2_coords[1] - pt1_coords[1])
                ax.add_patch(plt.Rectangle((lx, ly), w, h, fill=False))
                print("rect ")
                print(f"({lx}, {ly}), {w} {h}")
                pt1_coords = None

                text = input("What is the name of this area?")
                area_list.append([lx, ly, w, h, text])

                df = pandas.DataFrame(area_list)
                df.to_csv(output_path)

                fig.canvas.draw()

    plt.connect("button_press_event", on_click)

    plt.show()
