clear all
path='/home/bsl/py-faster-rcnn-master/tools/../lib/datasets/../../data/VOCdevkit2007'
comp_id='comp4-31647'
test_set='test'
output_dir='/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_test/ZF_faster_rcnn_final'
rm_res=1
res = voc_eval(path, comp_id, test_set, output_dir, rm_res)
