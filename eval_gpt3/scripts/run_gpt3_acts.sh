GAMENAME=${1}
LOG_FOLDER="xtx_${GAMENAME}"
GAME="${GAMENAME}.z5"
SEED=0

python3 -m eval_gpt3.gpt3_act --rom_path games/${GAME} \
                    --seed ${SEED} \
                    --bottleneck_directory ${2} \
                    --config_dir ${3} \
                    --project_name xtx-gpt3