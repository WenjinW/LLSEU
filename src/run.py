import sys, os, argparse, time
import logging
import json
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from dataloaders.my_dataset import MyDataset

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='cifar10', type=str,
                    choices=['pmnist', 'mixture', 'cifar10', 'cifar100'],
                    help='(default=%(default)s)')
parser.add_argument('--approach', default='seu', type=str,
                    choices=['ewc', 'ltg', 'progressive', 'seu'],
                    help='(default=%(default)s)')
# mode: training or search the best hyper-parameter
parser.add_argument('--mode', default='train', type=str, required=False, choices=['train', 'search'],
                    help='(default=%(default)s)')
# if debug is true, only use a small dataset
parser.add_argument('--debug', default='False', type=str, required=False, choices=['False', 'True'],
                    help='(default=%(default)s)')
parser.add_argument('--location', default='local', type=str, required=False, choices=['local', 'polyaxon'],
                    help='(default=%(default)s)')
# model: the basic model
parser.add_argument('--model',default='auto', type=str, required=False, choices=['alexnet', 'resnet', 'mlp', 'auto'],
                    help='(default=%(default)s)')
parser.add_argument('--output',default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--device', type=str, default='0', help='choose the device')
parser.add_argument('--id', type=str, default='0', help='the id of experiment')
parser.add_argument('--search_layers', default=6, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--eval_layers', default=20, type=int, required=False, help='(default=%(default)d)')
# hyper parameters in cell search stage
parser.add_argument('--c_epochs', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--c_batch', default=256, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--c_lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--c_lr_a', default=0.01, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--c_lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')

# hyper parameters in operation search stage
parser.add_argument('--o_epochs', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--o_batch', default=256, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--o_lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lr_a', default=0.01, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lamb_a', default=0.0003, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_lamb_size', default=1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--o_size', default=0, type=int, required=False,
help="the initial number of epochs for previous EUs")

# hyper parameters in training stage
parser.add_argument('--epochs', default=50, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--lamb_ewc', default=10000, type=float, required=False, help='(default=%(default)f)')

parser.add_argument('--lr_patience', default=5, type=int, required=False, help="Learning rate patience")
parser.add_argument('--lr_factor', default=0.3, type=float, required=False, help="Learning rate patience")

args = parser.parse_args()

if args.output == '':
    args.output = '../res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+args.id+'.txt'


print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args, arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:'+args.device)
else:
    device = torch.device('cpu')
    print('[CUDA unavailable]')
    sys.exit()

# Args -- Experiment
if args.experiment == 'pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment == 'cifar10':
    from dataloaders import cifar10 as dataloader
elif args.experiment == 'cifar100':
    from dataloaders import cifar100 as dataloader
elif args.experiment == 'mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'ltg':
    from approaches import ltg as approach
elif args.approach == 'progressive':
    from approaches import progressive as approach
elif args.approach == 'seu':
    from approaches import seu as approach

# Args -- Network
if args.model == 'resnet':
    if args.approach == 'ltg':
        from models import resnet_ltg as network
    else:  # ewc
        from models import resnet as network
elif args.model == 'alexnet':
    if args.approach == 'ltg':
        from models import alexnet_ltg as network
    elif args.approach == 'progressive':
        from models import alexnet_progressive as network
    else:  # ewc
        from models import alexnet as network
elif args.model == 'auto':
    network = None


########################################################################################################################
# define logger
logger = logging.getLogger()

# define tensorboard writer
exp_name = args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+args.id

# in polyaxon
if args.location == 'polyaxon':
    from polyaxon_client.tracking import Experiment
    experiment = Experiment()
    output_path = experiment.get_outputs_path()
    print("Output path: {}".format(output_path))
    logger.info("Output path: {}".format(output_path))
    writer = SummaryWriter(log_dir='/'+output_path)
else:
    writer = SummaryWriter(log_dir='../logs/' + exp_name)

# Load date
print('Load data...')
if args.location == 'polyaxon':
    data, taskcla, inputsize = dataloader.get(path='/plx-data/wjwang/', seed=args.seed)
else:
    data, taskcla, inputsize = dataloader.get(path='../dat/', seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)
logger.info('Input size =', inputsize, '\nTask info =', taskcla)


# logging the experiment config
config_exp = ["name: {}".format(exp_name),
              "mode: {}".format(args.mode),
              "dataset: {}".format(args.experiment),
              "approach: {}".format(args.approach),
              "device: {}".format(args.device),
              "id: {}".format(args.id),
              "task info: {}".format(taskcla),
              "Input size: {}".format(inputsize)]
for i, string in enumerate(config_exp):
    writer.add_text("Config_Experiment", string, i)
# logging the hyperparameter config
config_hyper_train = [
    "train_epochs: {}".format(args.epochs),
    "train_batch_size: {}".format(args.batch),
    "train_learning_rate: {}".format(args.lr),
    "train_weight_decay: {}".format(args.lamb),
    "learning_rate_patience: {}".format(args.lr_patience),
    "learning_factor: {}".format(args.lr_factor),
]

config_hyper_operation = [
    "operation_search_epochs: {}".format(args.o_epochs),
    "operation_search_batch_size: {}".format(args.o_batch),
    "operation_search_learning_rate_m: {}".format(args.o_lr),
    "operation_search_learning_rate_p: {}".format(args.o_lr_a),
    "operation_search_weight_decay_m: {}".format(args.o_lamb),
    "operation_search_weight_decay_p: {}".format(args.o_lamb_a),
    "operation_search_initial_epochs: {}".format(args.o_size)
]

config_hyper_cell = [
    "cell_search_epochs: {}".format(args.c_epochs),
    "cell_search_batch_size: {}".format(args.c_batch),
    "cell_search_learning_rate_m: {}".format(args.c_lr),
    "cell_search_learning_rate_p: {}".format(args.c_lr_a),
    "cell_search_weight_decay_m: {}".format(args.c_lamb),
]

for i, string in enumerate(config_hyper_train):
    writer.add_text("Config_Train", string, i)

for i, string in enumerate(config_hyper_operation):
    writer.add_text("Config_Operation", string, i)

for i, string in enumerate(config_hyper_cell):
    writer.add_text("Config_Cell", string, i)

# Inits
print('Inits...')
appr = None
if args.approach in ['ewc', 'progressive']:
    net = network.Net(inputsize, taskcla).to(device=device)
    utils.print_model_report(net)
    appr = approach.Appr(net, lr=args.lr, device=device, writer=writer, exp_name=exp_name,
                         args=args)
elif args.approach == 'ltg':
    net = network.Net(inputsize, taskcla).to(device=device)
    utils.print_model_report(net)
    appr = approach.Appr(net, lr=args.lr, device=device, writer=writer, exp_name=exp_name,
                         args=args)
elif args.approach == 'seu':
    appr = approach.Appr(input_size=inputsize, taskcla=taskcla, lr=args.lr, device=device,
                         writer=writer, exp_name=exp_name, args=args)


# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
model_size = []

for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*'*100)

    # get dataset
    train_data = MyDataset(data[t]['train'], debug=args.debug)
    valid_data = MyDataset(data[t]['valid'], debug=args.debug)

    # Train
    appr.train(t, train_data, valid_data, device=device)
    print('-'*100)

    # Test
    for u in range(t+1):

        test_data = MyDataset(data[u]['test'], debug=args.debug)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=4)

        test_loss, test_acc = appr.eval(u, test_loader, mode='train', device=device)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(
            u, data[u]['name'], test_loss, 100*test_acc))
        # experiment.log_metrics(task=u, step=t, test_loss=test_loss, test_acc=test_acc)
        writer.add_scalars('Test/Loss',
                           {'task{}'.format(u): test_loss}, global_step=t)
        writer.add_scalars('Test/Accuracy',
                           {'task{}'.format(u): test_acc * 100}, global_step=t)

        acc[t, u] = test_acc
        lss[t, u] = test_loss

    model_size.append(utils.get_model_size(appr.model, mode='M'))
    writer.add_scalars('ModelParameter(M)',
                       {'ModelParameter(M)': utils.get_model_size(appr.model, 'M')},
                       global_step=t)
    # Save
    # if args.location == 'polyaxon':
    #     np.savetxt("/"+output_path+"/"+exp_name+'.txt', acc, '%.4f')
    # else:
    #     np.savetxt("../res/" + exp_name + '.txt', acc, '%.4f')

# Done, logging the experiment results
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        writer.add_text("Results/Acc", '{:5.1f}% '.format(100*acc[i, j]), i)
        print('{:5.1f}% '.format(100*acc[i, j]), end='')
    print()
print('*'*100)
print('Done!')

writer.add_text("Results/Time", '[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
writer.add_text("Results/Mean_Acc", "Mean_Acc = {}".format(np.mean(acc[-1]).tolist()))
writer.add_text("Results/Size", "Size at last = {}".format(model_size[-1]))
# store the result of experiment
result = {"model_size": model_size, "mean_acc": np.mean(acc[-1]).tolist(), "accuracy": acc.tolist(),
          "elapsed_time(h)": (time.time()-tstart)/(60*60)}
if args.location == 'polyaxon':
    with open("/" + output_path + "/" + exp_name + '.json', 'w') as f:
        json.dump(result, f)
else:
    with open("../res/" + exp_name + '_' + str(id) + '.json', 'w') as f:
        json.dump(result, f)
print("Results/Time", '[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
