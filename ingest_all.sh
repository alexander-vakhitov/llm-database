base_dir="/media/alexander/slamcore_data/dhl/shift_trajectories/filtered_shift_trajectories/"
zones="/media/alexander/slamcore_data/dhl/floorplan_annotation/area.csv"
floorplan="/media/alexander/slamcore_data/dhl/Daventry_Oct24_scaled.png"
session="/media/alexander/slamcore_data/dhl/dhl_jdw_map04_aligned_racks.session"
for forklift_id in 1 4 5 6 7 8 9; do
    for day in "25" "26" "27" "28" "29"; do
        echo ${day}
        if [[ $forklift_id == 1 || $forklift_id == 6 ]]; then
          python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_0600.csv ${forklift_id} day $zones $floorplan $session
          python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_1800.csv ${forklift_id} night $zones $floorplan $session
        else
          python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_0600_cleaned_cleaned.csv ${forklift_id} day $zones $floorplan $session
          python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_1800_cleaned_cleaned.csv ${forklift_id} night $zones $floorplan $session
        fi
    done
done
