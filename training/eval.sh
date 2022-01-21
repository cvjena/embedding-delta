#!/bin/sh


sh_root=/data/Uni/compling/stss/stss-2021-eval-ps
predictions_folder=/data/Uni/compling/stss/stss-2021-eval-ps/predictions
config_base_folder=/data/Uni/compling/stss/ps_json
results_base=/data/Uni/compling/stss/ps_results

counter=0
num_files=`ls $config_base_folder | wc -l`
mapfile -t dirlist < <( find $config_base_folder  -mindepth 1 -type d -printf '%f\n' )
for current_config_folder in ${dirlist[@]} ; do
    let counter++
    echo "current file is "$current_config_folder", "$counter" of "$num_files
    config_full=$config_base_folder"/"$current_config_folder
    rm -rf $predictions_folder
    ln -s $config_full $predictions_folder
    cd $sh_root
    results_filename=$results_base"/"$current_config_folder
    sh run.sh | grep -v / | grep -v \( > $results_filename
done




