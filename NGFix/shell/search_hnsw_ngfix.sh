MEX=48
M=16
efC=500
efC_AKNN=1500
taskset -c 1 test/search_hnsw_ngfix \
--test_query_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
--test_gt_path /workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin \
--metric ip_float --K 100 --result_path /workspace/NGFix/data/t2i-10M/test_t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.csv \
--index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index \

echo "search_hnsw_ngfix_aknn done, now start search_hnsw_bottom"

taskset -c 1 test/search_hnsw_ngfix \
--test_query_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
--test_gt_path /workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin \
--metric ip_float --K 100 --result_path /workspace/NGFix/data/t2i-10M/test_t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.csv \
--index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \

# MEX=48
# M=16
# efC=500
# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/MainSearch/mainse_query_test.fbin \
# --test_gt_path /SSD/MainSearch/mainse_query_test_gt.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/mainse_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \


# MEX=48
# M=16
# efC=500
# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \

# MEX=48
# M=16
# efC=500
# taskset -c 1 test/search_hnsw_ngfix \
# --test_query_path /SSD/SIFT1M/sift_query.fbin \
# --test_gt_path /SSD/SIFT1M/gt.query.top100.bin \
# --metric l2_float --K 100 --result_path /home/hzy/NGFix/result/test_t2i.csv \
# --index_path /SSD/models/NGFix/sift1M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
