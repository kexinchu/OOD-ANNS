# use Text-to-Image 8M to build index first 
MEX=48
M=16
efC=500
efC_AKNN=1500
./test/build_hnsw_bottom --base_data_path /workspace/RoarGraph/data/t2i-10M/base.8M.fbin \
--metric ip_float --train_number ${train_number} \
--result_hnsw_index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_8M.index \
--M ${M} --MEX ${MEX} --efC ${efC}

./test/build_hnsw_ngfix_aknn \
--train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
--train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
--base_graph_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_8M.index \
--metric ip_float --efC_AKNN ${efC_AKNN} \
--result_index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_8M.index \


now we insert 2M data
./test/test_hnsw_ngfix_insertion --base_data_path /workspace/RoarGraph/data/t2i-10M/base.10M.fbin \
--train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
--train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
--raw_index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_8M.index \
--metric ip_float \
--result_index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_Insert2M_8M.index \
--efC ${efC} --insert_st_id 8000000\

taskset -c 1 test/search_hnsw_ngfix \
--test_query_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
--test_gt_path /workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin \
--metric ip_float --K 100 --result_path /workspace/NGFix/data/t2i-10M/test_t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_Insert2M_8M.csv \
--index_path /workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_Insert2M_8M.index \




