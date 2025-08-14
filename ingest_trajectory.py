import sqlite3
import os
import argparse
import pandas
import numpy as np
import datetime
from util import get_shift_time, get_activity_periods, get_velocities


if __name__ == '__main__':
    time_subsampling_rate = 15
    min_time_interval_ns = 600 * 1e9
    
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("forklift_id", type=int)
    parser.add_argument("shift_type", choices=['day', 'night'])
    
    args = parser.parse_args()
    db_path = "aware_data.db"
    is_creating_tables = not os.path.exists(db_path)    
    
    con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES |
                                                sqlite3.PARSE_COLNAMES)
    cur = con.cursor()
    if is_creating_tables:
        cur.execute("CREATE TABLE activity(id int primary key, forklift_id int, start timestamp, end timestamp, status int)")        
        con.commit()
        
    forklift_id = args.forklift_id
    
    df = pandas.read_csv(args.csv_path, dtype={"acq_timestamp [ns]": np.int64})        
    
    # data processing params
    time_subsampling_rate = 15
    static_threshold = 0.05
    # data processing starts here
    velocities, velocity_timestamps, data_mask = get_velocities(df, time_subsampling_rate)
    
    static_mask = velocities < static_threshold
    dynamic_mask = np.invert(static_mask)
    
    activity_timestamps = velocity_timestamps[dynamic_mask]
            
    shift_start, shift_end = get_shift_time(velocity_timestamps, args.shift_type)    
    time_periods = get_activity_periods(min_time_interval_ns, shift_start, shift_end, activity_timestamps)
    print(len(time_periods))
    query_with_param = "INSERT INTO activity(forklift_id, start, end, status) VALUES(?, ?, ?, ?)"
    for activity_record in time_periods:
        time_start, time_end, status = activity_record
        time_start = datetime.datetime.fromtimestamp(int(time_start / 1e9))
        time_end = datetime.datetime.fromtimestamp(int(time_end / 1e9))
        cur.execute(query_with_param, (forklift_id, time_start, time_end, status))
    con.commit()
    # database filled