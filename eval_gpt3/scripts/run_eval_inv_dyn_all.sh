for game_dir in bottleneck_state_cache/*/ ; do
    game_name=$(basename ${game_dir})
    for bottleneck_dir in ${game_dir}*/ ; do
        for file in ${bottleneck_dir}* ; do
            if [[ "$file" == *"gpt3acts"* ]]; then
                echo "Starting Evaluation for Game ${game_name} and GPT3 Action File ${file}..."
                for i in 0 1 2 3 4 5 6 7 8 9 10 ; do
                    ./eval_gpt3/scripts/run_eval_inv_dyn.sh ${game_name} ${bottleneck_dir} ${file} $i
                done
            fi
        done
    done
done