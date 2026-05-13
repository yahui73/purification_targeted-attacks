# packages
# software packages
import torch
import numpy as np
import traceback
import argparse
import yaml
import logging
import time
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import sys
sys.path.append('/home/lyh/Desktop/TA-DCH/stablediffpuri/')
from runners import *   # runner wenjianjia

# main.py
# (1) Import configs (replace argparse. Too many argparse flags now)
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # Dataset and save logs
    parser.add_argument('--log', default='imgs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='WIKI_default.yml',
                        help='Path for saving running related data.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    parser.add_argument('--exp_mode', type=str, default='Full', help='Available: [Full, Partial, One]')
    parser.add_argument('--runner', type=str, default='Empirical', help='Available: [Empirical, Certified, Deploy]')

    # Arguments not to be touched
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    # parser.add_argument('--CIFARC_CLASS', type=int, default=-1)
    # parser.add_argument('--CIFARC_SEV', type=int, default=1)

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])

    # parse config file
    with open(os.path.join('/home/lyh/Desktop/TA-DCH/stablediffpuri/configs/', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)

    # define the folder name
    if new_config.purification.cond:
        args.log = os.path.join("logs", "{}_{}_COND:{}".format(
            new_config.structure.dataset,
            str(new_config.attack.attack_method),
            new_config.purification.guide_mode
        ),
                                "step_{}_iter_{}_path_{}_per={}_{}".format(
                                    new_config.purification.purify_step,
                                    new_config.purification.max_iter,
                                    new_config.purification.path_number,
                                    new_config.attack.ptb,
                                    f'{new_config.purification.guide_scale}+{new_config.purification.guide_scale_base}'
                                ))
    else:
        args.log = os.path.join("logs", "{}_{}".format(
            new_config.structure.dataset,
            str(new_config.attack.attack_method)
        ),
                                "step_{}_iter_{}_path_{}_per={}".format(
                                    new_config.purification.purify_step,
                                    new_config.purification.max_iter,
                                    new_config.purification.path_number,
                                    new_config.attack.ptb
                                ))
    # create folder
    # if os.path.exists(args.log):
    #     shutil.rmtree(args.log)
    if not os.path.exists(args.log):
        os.makedirs(args.log, exist_ok=True)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config

def parse_args_and_config_step():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # Dataset and save logs
    parser.add_argument('--log', default='imgs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='WIKI_default_step.yml',
                        help='Path for saving running related data.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    parser.add_argument('--exp_mode', type=str, default='Full', help='Available: [Full, Partial, One]')
    parser.add_argument('--runner', type=str, default='Empirical', help='Available: [Empirical, Certified, Deploy]')

    # Arguments not to be touched
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    # parser.add_argument('--CIFARC_CLASS', type=int, default=-1)
    # parser.add_argument('--CIFARC_SEV', type=int, default=1)

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])

    # parse config file
    with open(os.path.join('/home/lyh/Desktop/TA-DCH/stablediffpuri/configs/', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)

    # define the folder name
    if new_config.purification.cond:
        args.log = os.path.join("logs", "{}_{}_COND:{}".format(
            new_config.structure.dataset,
            str(new_config.attack.attack_method),
            new_config.purification.guide_mode
        ),
                                "step_{}_iter_{}_path_{}_per={}_{}".format(
                                    new_config.purification.purify_step,
                                    new_config.purification.max_iter,
                                    new_config.purification.path_number,
                                    new_config.attack.ptb,
                                    f'{new_config.purification.guide_scale}+{new_config.purification.guide_scale_base}'
                                ))
    else:
        args.log = os.path.join("logs", "{}_{}".format(
            new_config.structure.dataset,
            str(new_config.attack.attack_method)
        ),
                                "step_{}_iter_{}_path_{}_per={}".format(
                                    new_config.purification.purify_step,
                                    new_config.purification.max_iter,
                                    new_config.purification.path_number,
                                    new_config.attack.ptb
                                ))
    # create folder
    # if os.path.exists(args.log):
    #     shutil.rmtree(args.log)
    if not os.path.exists(args.log):
        os.makedirs(args.log, exist_ok=True)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config

def parse_args_and_config_step_30():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # Dataset and save logs
    parser.add_argument('--log', default='imgs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='WIKI_step_30.yml',
                        help='Path for saving running related data.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    parser.add_argument('--exp_mode', type=str, default='Full', help='Available: [Full, Partial, One]')
    parser.add_argument('--runner', type=str, default='Empirical', help='Available: [Empirical, Certified, Deploy]')

    # Arguments not to be touched
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    # parser.add_argument('--CIFARC_CLASS', type=int, default=-1)
    # parser.add_argument('--CIFARC_SEV', type=int, default=1)

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])

    # parse config file
    with open(os.path.join('/home/lyh/Desktop/TA-DCH/stablediffpuri/configs/', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)

    # define the folder name
    if new_config.purification.cond:
        args.log = os.path.join("logs", "{}_{}_COND:{}".format(
            new_config.structure.dataset,
            str(new_config.attack.attack_method),
            new_config.purification.guide_mode
        ),
                                "step_{}_iter_{}_path_{}_per={}_{}".format(
                                    new_config.purification.purify_step,
                                    new_config.purification.max_iter,
                                    new_config.purification.path_number,
                                    new_config.attack.ptb,
                                    f'{new_config.purification.guide_scale}+{new_config.purification.guide_scale_base}'
                                ))
    else:
        args.log = os.path.join("logs", "{}_{}".format(
            new_config.structure.dataset,
            str(new_config.attack.attack_method)
        ),
                                "step_{}_iter_{}_path_{}_per={}".format(
                                    new_config.purification.purify_step,
                                    new_config.purification.max_iter,
                                    new_config.purification.path_number,
                                    new_config.attack.ptb
                                ))
    # create folder
    # if os.path.exists(args.log):
    #     shutil.rmtree(args.log)
    if not os.path.exists(args.log):
        os.makedirs(args.log, exist_ok=True)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    print('')
    args, config = parse_args_and_config()
    log_progress = open(os.path.join(args.log, f"log_progress_{config.device.rank}"), "w")
    # sys.stdout = log_progress #assignent(lyh)
    logging.info("Config =")
    print(">" * 80)
    print(config)
    print("<" * 80)

    try:
        runner = eval(args.runner)(args, config)
        runner.run(log_progress)
    except:
        logging.error(str(traceback.format_exc()))

    log_progress.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
