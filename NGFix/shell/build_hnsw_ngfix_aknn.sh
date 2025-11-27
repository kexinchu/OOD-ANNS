MEX=48
M=16
efC=500
efC_AKNN=1500
./test/build_hnsw_ngfix_aknn \
--train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
--train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
--base_graph_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
--metric ip_float --efC_AKNN ${efC_AKNN} \
--result_index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index \


# MEX=48
# M=16
# efC=500
# efC_AKNN=1500
# ./test/build_hnsw_ngfix_aknn \
# --train_query_path /SSD/MainSearch/train.1M.fbin \
# --train_gt_path /SSD/MainSearch/gt.train.1M.bin \
# --base_graph_path /SSD/models/NGFix/mainse_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --metric ip_float --efC_AKNN ${efC_AKNN} \
# --result_index_path /SSD/models/NGFix/mainse_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index \

