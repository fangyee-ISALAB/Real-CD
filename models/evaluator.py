import os
import numpy as np
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import torch.optim as optim
import matplotlib.pyplot as plt
from helper.registry import ADAPTATION_REGISTRY
from tta_methods.tent import Tent
from misc.inference_profiler import InferenceProfiler


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader
        self.args = args
        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir


        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


        # if "SYSU" in self.args.data_name:
        #     self.cnt = 10
        # else:
        #     self.cnt = 1
        self.cnt = 1
        self.feat_collect = []

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'], strict=True)
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """

        target = self.batch['L'].to(self.device).detach()

        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        batch_size = self.batch['L'].shape[0]
        for i in range(batch_size):
            # 处理第 i 个样本
            vis_gt = utils.make_numpy_grid(self.batch['L'][i])
            vis_pred = utils.make_numpy_grid(self._visualize_pred()[i])  # 假设返回整个 batch 的预测
            vis_color = np.zeros_like(vis_gt)

            # 数值截断
            vis_gt = np.clip(vis_gt, a_min=0.0, a_max=1.0)
            vis_pred = np.clip(vis_pred, a_min=0.0, a_max=1.0)

            # 颜色标注规则
            vis_color[(vis_pred[:, :, 0] == 1) & (vis_gt[:, :, 0] == 0)] = [1, 0, 0]  # 假阳性（红色）
            vis_color[(vis_pred[:, :, 0] == 0) & (vis_gt[:, :, 0] == 1)] = [0, 0, 1]  # 假阴性（蓝色）
            vis_color[(vis_pred[:, :, 0] == 1) & (vis_gt[:, :, 0] == 1)] = [1, 1, 1]  # 真阳性（白色）

            vis_color = np.clip(vis_color, a_min=0.0, a_max=1.0)

            # 生成带索引的唯一文件名
            file_name_pred = os.path.join(
                self.vis_dir, f'eval_{self.batch["name"][i][:-4]}_pred_{i}.png')
            file_name_color = os.path.join(
                self.vis_dir, f'eval_{self.batch["name"][i][:-4]}_pred_color_{i}.png')
            file_name_gt = os.path.join(
                self.vis_dir, f'eval_{self.batch["name"][i][:-4]}_gt_{i}.png')

            # 保存所有图像
            # plt.imsave(file_name_pred, vis_pred)
            plt.imsave(file_name_color, vis_color)
            # plt.imsave(file_name_gt, vis_gt)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']
        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):

        self.batch = batch
        img_in1 = self.batch['A'].to("cuda")
        img_in2 = self.batch['B'].to("cuda")
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

    def eval_models(self, cfg, checkpoint_name='best_ckpt.pt'):
        print("==================")
        print(self.checkpoint_dir)
        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()


        # Iterate over data.
        self.logger.write(self.args.data_name)
        self.net_G = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=self.net_G, num_classes=2)
        self.logger.write(f'\nTest time ADAPTATION is {cfg.MODEL.ADAPTATION}\n')

        profiler = InferenceProfiler()
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            profiler.start_timer()
            with torch.no_grad():
                self._forward_pass(batch)
                torch.cuda.empty_cache()
            profiler.stop_timer(batch_size=self.batch['L'].shape[0])
            self._collect_running_batch_states()
        profiler.summary(f"{self.args.data_name}_{cfg.MODEL.ADAPTATION}")
        self._collect_epoch_states()



