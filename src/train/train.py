import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger


parser = argparse.ArgumentParser(description='Uni-ControlNet Training')
parser.add_argument('--config-path', type=str, default='./configs/local_v15.yaml')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('---batch-size', type=int, default=4)
parser.add_argument('---training-steps', type=int, default=1e5)
parser.add_argument('---resume-path', type=str, default='./ckpt/init_local_4con.ckpt')
parser.add_argument('--geonet_path',type= str, default='/home/x.zhexiao/near-remote/src/geo-attention/outputbrooklyn-fc8_landuse_geo-attention/lightning_logs/version_2/checkpoints/last.ckpt' )
parser.add_argument('---logdir', type=str, default='./log_local/')
parser.add_argument('---log-freq', type=int, default=20)
parser.add_argument('---sd-locked', type=bool, default=True)
parser.add_argument('---num-workers', type=int, default=16)
parser.add_argument('---gpus', type=int, default=-1)

parser.add_argument('---num_local_conditions', type=int, default=4)

# geoattention params
parser.add_argument(
      '--context',
      default="distance,orientation,panorama,overhead",
      type=lambda s: {x: True for x in sorted(s.strip().split(','))},
      help=
      "Inputs for geospatial attention [distance, orientation, panorama, overhead]"
  )

args = parser.parse_args()


def main():

    config_path = args.config_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_steps = args.training_steps
    resume_path = args.resume_path
    default_logdir = args.logdir
    logger_freq = args.log_freq
    sd_locked = args.sd_locked
    num_workers = args.num_workers
    gpus = args.gpus
    num_local_conditions = args.num_local_conditions
    
    geonet_path = args.geonet_path
    config = OmegaConf.load(config_path)
    #import pdb; pdb.set_trace()
    model = instantiate_from_config(config['model'])

    #加geo attention的预训练
    model.load_state_dict(load_state_dict(geonet_path,location='cpu'),strict = False)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False) #忽略geonet的部分
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    dataset = instantiate_from_config(config['data'])
    # import pdb; pdb.set_trace()
    # dataset.getitem(1)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    #import pdb; pdb.set_trace()
    logger = ImageLogger(batch_frequency=logger_freq, num_local_conditions=num_local_conditions)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
    )
        
    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[logger, checkpoint_callback], 
        default_root_dir=default_logdir,
        max_steps=training_steps,
        #fast_dev_run=True       # for debug only
    )
    trainer.fit(model,
        dataloader, 
    )


if __name__ == '__main__':
    main()