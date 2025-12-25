#bin/bash

nohup ./test/test_runtime_update_end2end_ngfix \
    --base_index_path /workspace/RoarGraph/data/comparison_10M/base.index \
    --train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
    --train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
    --additional_vector_path /workspace/RoarGraph/data/t2i-10M/base.additional.10M.fbin \
    --metric ip_float \
    --result_dir /workspace/RoarGraph/data/runtime_update_test \
    --K 100 \
    --duration_minutes 60 \
    > /workspace/OOD-ANNS/Ours/data/runtime_update_test/nohup_ngfix.out 2>&1 &
