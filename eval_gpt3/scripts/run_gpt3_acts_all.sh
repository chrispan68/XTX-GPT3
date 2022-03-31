for game_dir in bottleneck_state_cache/*/ ; do
    game_name=$(basename ${game_dir})
    for bottleneck_dir in ${game_dir}*/ ; do
        ./eval_gpt3/scripts/run_gpt3_acts.sh ${game_name} ${bottleneck_dir} ${1}
    done
done