#!/bin/bash
# for num_spammer from 4 to 9, assign GPU (num_spammer-3) and run experiments in background
rm -f gpu_*.log
dataset_name="hotpotqa"
for num_src in {4..9}; do
    (
        # GPU 번호 지정 (예: num_src=4이면 GPU 1 사용)
        GPU_ID=$((num_src - 3))
        # Define the base directory (필요에 맞게 경로 수정)
        BASE_DIR="/hdd/hdd1/refined_contriever_results/0912_${dataset_name}_1600_normal_rag_incontext_num_src_${num_src}_p_rel_0.6_top_k_3_exp_iter_10"
        
        # 각 num_spammer에 대해 0부터 9까지 iteration 실행
        for i in {0..9}; do
            DATA_PATH="$BASE_DIR/$i/retrieval_results/baseline_for_robustrag.json"

            echo "Starting iteration $i for num_src=$num_src on GPU $GPU_ID at $(date)" >> gpu_${GPU_ID}.log

            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --dataset_name "$dataset_name" \
                --model_name llama7b \
                --top_k 10 \
                --attack_method none \
                --defense_method keyword \
                --corruption_size 0 \
                --no_vanilla \
                --save_response \
                --data_path "$DATA_PATH" \
                --num_src $num_src \
                --iteration "$i" \
                --num_spammer 0 >> gpu_${GPU_ID}.log 2>&1

            echo "Finished iteration $i for num_src=$num_src on GPU $GPU_ID at $(date)" >> gpu_${GPU_ID}.log
        done
    ) &
done

# 기다림: 백그라운드에 있는 모든 작업이 끝날 때까지 대기
wait
echo "All experiments finished."
