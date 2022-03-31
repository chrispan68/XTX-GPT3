#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=eval-gpt3-inv-dyn

# Remove one # to uncommment
#SBATCH --output=%x-%j.txt

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=2
#SBATCH --mem=11G
#SBATCH --time=1-23:59:00  
#SBATCH --gres=gpu:1

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=chrispan@princeton.edu


# Submit jobs.

version=4

conda activate drrn-shunyu

./eval_gpt3/scripts/run_eval_inv_dyn_all.sh

wait; #Make sure to wait till all the runs have completed.

# Finish the script
exit 0