base_dir="/media/alexander/slamcore_data/dhl/shift_trajectories/filtered_shift_trajectories/"
for forklift_id in 1 4 5 6 7 8 9; do
    for day in "25" "26" "27" "28" "29"; do
        echo ${day}
        echo "day"
        python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_0600.csv ${forklift_id} day
        python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_0600_cleaned_cleaned.csv ${forklift_id} day
        echo "night"
        python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_1800.csv ${forklift_id} night
        python ingest_trajectory.py ${base_dir}/${forklift_id}/${forklift_id}_shift_202411${day}_1800_cleaned_cleaned.csv ${forklift_id} night
    done
done