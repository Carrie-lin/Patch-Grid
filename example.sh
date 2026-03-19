## with shape space
python train.py --dataset_path data/train/00000006_0 --exp_name 00000006_0_ss --pretrained _results/shape_space/decoder_1.pth --fix_decoder --use_shape_space
python extract.py --pretrained _results/models/00000006_0_ss.pth --mesh_path_prefix _results/mesh --exp_name 00000006_0_ss --dataset_path data/train/00000006_0
python metrics/mesh_metrics.py --gt_path data/train/00000006_0/00000006_0.obj --recon_path _results/mesh/00000006_0_ss_composed_all.obj --metrics_file metrics.txt

python train.py --dataset_path data/train/00000089_1 --exp_name 00000089_1_ss --pretrained _results/shape_space/decoder_1.pth --fix_decoder --use_shape_space
python extract.py --pretrained _results/models/00000089_1_ss.pth --mesh_path_prefix _results/mesh --exp_name 00000089_1_ss --dataset_path data/train/00000089_1
python metrics/mesh_metrics.py --gt_path data/train/00000089_1/00000089_1.obj --recon_path _results/mesh/00000089_1_ss_composed_all.obj --metrics_file metrics.txt

python train.py --dataset_path data/train/00000008_0 --exp_name 00000008_0_ss --pretrained _results/shape_space/decoder_1.pth --fix_decoder --use_shape_space
python extract.py --pretrained _results/models/00000008_0_ss.pth --mesh_path_prefix _results/mesh --exp_name 00000008_0_ss --dataset_path data/train/00000008_0
python metrics/mesh_metrics.py --gt_path data/train/00000008_0/00000008_0.obj --recon_path _results/mesh/00000008_0_ss_composed_all.obj --metrics_file metrics.txt


## train from scratch
python train.py --dataset_path data/train/00000006_0 --exp_name 00000006_0_no_ss --epochs 500 --mile_stones 470 485
python extract.py --pretrained _results/models/00000006_0_no_ss.pth --mesh_path_prefix _results/mesh --exp_name 00000006_0_no_ss --dataset_path data/train/00000006_0
python metrics/mesh_metrics.py --gt_path data/train/00000006_0/00000006_0.obj --recon_path _results/mesh/00000006_0_no_ss_composed_all.obj --metrics_file metrics_no_ss.txt

python train.py --dataset_path data/train/00000089_1 --exp_name 00000089_1_no_ss --epochs 500 --mile_stones 470 485
python extract.py --pretrained _results/models/00000089_1_no_ss.pth --mesh_path_prefix _results/mesh --exp_name 00000089_1_no_ss --dataset_path data/train/00000089_1
python metrics/mesh_metrics.py --gt_path data/train/00000089_1/00000089_1.obj --recon_path _results/mesh/00000089_1_no_ss_composed_all.obj --metrics_file metrics_no_ss.txt

python train.py --dataset_path data/train/00000008_0 --exp_name 00000008_0_no_ss --epochs 500 --mile_stones 470 485
python extract.py --pretrained _results/models/00000008_0_no_ss.pth --mesh_path_prefix _results/mesh --exp_name 00000008_0_no_ss --dataset_path data/train/00000008_0
python metrics/mesh_metrics.py --gt_path data/train/00000008_0/00000008_0.obj --recon_path _results/mesh/00000008_0_no_ss_composed_all.obj --metrics_file metrics_no_ss.txt



