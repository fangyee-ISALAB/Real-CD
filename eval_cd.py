from argparse import ArgumentParser
from models.evaluator import *
from cfgs.conf import cfg, load_cfg_from_args
import argparse
print(torch.cuda.is_available())


"""
eval the CD model
"""

def main(description):
    # ------------
    # args
    # ------------

    parser = ArgumentParser(description)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--checkpoints_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_root', default='data', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)
    parser.add_argument('--tta_method', default='', type=str)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")

    args = parser.parse_args()
    load_cfg_from_args(args)

    utils.get_device(args)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join("./visualization", cfg.MODEL.ADAPTATION, args.data_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    print(args.data_root)
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split, data_root=args.data_root)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(cfg, checkpoint_name=args.checkpoint_name)



if __name__ == '__main__':
    main('Evaluation.')
    torch.cuda.empty_cache()

