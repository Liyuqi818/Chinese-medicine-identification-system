import os
import sys
import argparse
import basetrainer
from basetrainer.engine import trainer
from basetrainer.engine.launch import launch
from basetrainer.metric import accuracy_recorder
from basetrainer.callbacks import log_history, model_checkpoint, losses_recorder
from basetrainer.scheduler import build_scheduler
from basetrainer.optimizer.build_optimizer import get_optimizer
from basetrainer.utils import log, file_utils, setup_config, torch_tools
from classifier.criterion.build_criterion import get_criterion
from classifier.models import build_models
from classifier.dataloader import build_dataset
from classifier.transforms.build_transform import image_transform

sys.path.append(os.getcwd())

print(basetrainer.__version__)

class ClassificationTrainer(trainer.EngineTrainer):
    """ Training Pipeline """

    def __init__(self, cfg):
        super(ClassificationTrainer, self).__init__(cfg)
        torch_tools.set_env_random_seed()
        time = file_utils.get_time()
        cfg.work_dir = os.path.join(cfg.work_dir, "_".join([cfg.net_type, str(cfg.width_mult), cfg.loss_type, time]))
        cfg.model_root = os.path.join(cfg.work_dir, "model")
        cfg.log_root = os.path.join(cfg.work_dir, "log")
        if self.is_main_process:
            file_utils.create_dir(cfg.work_dir)
            file_utils.create_dir(cfg.model_root)
            file_utils.create_dir(cfg.log_root)
            file_utils.copy_file_to_dir(cfg.config_file, cfg.work_dir)
            setup_config.save_config(cfg, os.path.join(cfg.work_dir, "setup_config.yaml"))
        self.logger = log.set_logger(level="debug",
                                     logfile=os.path.join(cfg.log_root, "train.log"),
                                     is_main_process=self.is_main_process)
        # build project
        self.build(cfg)
        self.logger.info("=" * 60)
        self.logger.info("work_dir          :{}".format(cfg.work_dir))
        self.logger.info("config_file       :{}".format(cfg.config_file))
        self.logger.info("gpu_id            :{}".format(cfg.gpu_id))
        self.logger.info("main device       :{}".format(self.device))
        self.logger.info("num_classes       :{}".format(cfg.num_classes))
        self.logger.info("num_samples(train):{}".format(self.train_num))
        self.logger.info("mean_num(train)   :{}".format(self.train_num // cfg.num_classes))
        self.logger.info("num_samples(test) :{}".format(self.test_num))
        self.logger.info("mean_num(test)    :{}".format(self.test_num // cfg.num_classes))
        self.logger.info("loss_type         :{}".format(cfg.loss_type))
        self.logger.info("=" * 60)

    # ... [Other methods of ClassificationTrainer class] ...

# ... [Other function definitions] ...

def main(cfg):
    t = ClassificationTrainer(cfg)
    return t.run()

def get_parser():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    # Removed the default value of config_file argument
    parser.add_argument("-c", "--config_file", help="configs file", type=str)
    parser.add_argument('--distributed', action='store_true', help='use distributed training', default=False)
    return parser

if __name__ == "__main__":
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    # Get the directory of the current script
    current_dir = os.path.dirname(current_script_path)
    # Build the absolute path of the config.yaml file
    config_path = os.path.join(current_dir, "configs", "config.yaml")

    # Initialize the parser and set the default config file path
    parser = get_parser()
    parser.set_defaults(config_file=config_path)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set up the configuration using the parsed arguments
    cfg = setup_config.parser_config(args, cfg_updata=True)

    # Start the training process
    launch(main,
           num_gpus_per_machine=len(cfg.gpu_id),
           dist_url="tcp://127.0.0.1:28661",
           num_machines=1,
           machine_rank=0,
           distributed=cfg.distributed,
           args=(cfg,))
