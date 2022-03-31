GAMENAME=${1}
LOG_FOLDER="xtx_${GAMENAME}"
GAME="${GAMENAME}.z5"
SEED=0
JERICHO_SEED=$SEED # set to -1 if you want stochastic version
MODEL_NAME='inv_dy'
JERICHO_ADD_WT='add_wt' # change to 'no_add_wt' if you don't want to add extra actions to game grammar

python3 -m eval_gpt3.eval_gpt3 --output_dir logs/${LOG_FOLDER} \
                    --rom_path games/${GAME} \
                    --seed ${SEED} \
                    --jericho_seed ${JERICHO_SEED} \
                    --model_name ${MODEL_NAME} \
                    --eval_freq 10000000 \
                    --memory_size 10000 \
                    --w_inv 1 \
                    --r_for 1 \
                    --w_act 1 \
                    --jericho_add_wt ${JERICHO_ADD_WT} \
                    --bottleneck_directory ${2} \
                    --num_envs 10 \
                    --gpt3_acts_filename ${3} \
                    --gpt3_steps ${4} \
                    --output_dir ${2}logs/logs