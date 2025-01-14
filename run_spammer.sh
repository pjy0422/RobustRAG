#!/bin/bash
# for num_spammer from 1 to 7, assign GPU (num_spammer-1) and run experiments in background
for num_spammer in {1..7}; do
    (
        # GPU 번호 지정 (예: num_spammer=1이면 GPU 0 사용)
        GPU_ID=$((num_spammer - 1))
        # Define the base directory (필요에 맞게 경로 수정)
        BASE_DIR="/home/guest-pjy/spammer_hammer/0912_NQ_1600_adv_hammer_${num_spammer}adv_rag_incontext_num_src_9_p_rel_0.6_top_k_3_exp_iter_10"
        
        # 각 num_spammer에 대해 0부터 9까지 iteration 실행
        for i in {0..9}; do
            DATA_PATH="$BASE_DIR/$i/retrieval_results/baseline_for_robustrag.json"

            echo "Starting iteration $i for num_spammer=$num_spammer on GPU $GPU_ID at $(date)" >> gpu_${GPU_ID}.log

            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --model_name llama7b \
                --top_k 10 \
                --attack_method none \
                --defense_method keyword \
                --corruption_size 0 \
                --no_vanilla \
                --save_response \
                --data_path "$DATA_PATH" \
                --num_src 9 \
                --iteration "$i" \
                --num_spammer "$num_spammer" >> gpu_${GPU_ID}.log 2>&1

            echo "Finished iteration $i for num_spammer=$num_spammer on GPU $GPU_ID at $(date)" >> gpu_${GPU_ID}.log
        done
    ) &
done

# 기다림: 백그라운드에 있는 모든 작업이 끝날 때까지 대기
wait
echo "All experiments finished."
