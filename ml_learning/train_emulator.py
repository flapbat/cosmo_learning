import torch
import numpy as np
import os
import sys
from datetime import datetime
from emulator import ResTRF, ResMLP
import yaml
import h5py as h5
import argparse

#===================================================================================================
# Command line args

parser = argparse.ArgumentParser(prog='train_emulator')

parser.add_argument("--yaml", "-y",
    dest="cobaya_yaml",
    help="The training YAML containing the training_args block",
    type=str,
    nargs='?')

parser.add_argument("--probe", "-p",
    dest="probe",
    help="the probe, listed in the yaml, of which to generate data vectors for.",
    type=str,
    nargs='?')

parser.add_argument("--epochs", "-e",
    dest="n_epochs",
    help="(int) number of training epochs. Default=250",
    type=int,
    default=250,
    nargs='?')

parser.add_argument("--batchsize", "-b",
    dest="batch_size",
    help="(int) batch size to use while training. Default=256",
    type=int,
    default=256,
    nargs='?')

parser.add_argument("--learning_rate", "-lr",
    dest="learning_rate",
    help="(float) learning rate to use while training. Default=1e-3",
    type=float,
    default=1e-3,
    nargs='?')

parser.add_argument("--weight_decay", "-wd",
    dest="weight_decay",
    help="(float) Weight decay (adds L2 norm of model weights to loss fcn) to use while training. Default=0",
    type=float,
    default=0.0,
    nargs='?')

parser.add_argument("--save_losses", "-s",
    dest="save_losses",
    help="(bool) Save losses at each epoch to a text file 'losses.txt'. Default=False",
    type=bool,
    default=False,
    nargs='?')
        
# ======== Additions by Béla =======  
parser.add_argument("--save_testing_metrics", "-stm",
    dest="save_testing_metrics",
    help="(bool) Save testing metrics (chi2 mean, median, fractions) at each epoch to 'testing_metrics.txt'. Default=False",
    type=bool,
    default=False,
    nargs='?')

parser.add_argument("--squeeze_factor", "-sf",
    dest="squeeze_factor",
    help="(float) Factor to divide covariance matrix by (cov = cov/squeeze_factor). Default=1.0 (no squeezing)",
    type=float,
    default=1.0,
    nargs='?')

# === TRANSFER LEARNING ARGUMENTS ===
parser.add_argument("--transfer_learning", "-tl",
    dest="transfer_learning",
    help="(bool) Enable transfer learning mode. Default=False",
    type=bool,
    default=False,
    nargs='?')

parser.add_argument("--pretrained_model", "-pm",
    dest="pretrained_model",
    help="Path to pretrained model (.pth file) for transfer learning",
    type=str,
    default=None,
    nargs='?')

parser.add_argument("--freeze_strategy", "-fs",
    dest="freeze_strategy",
    help="Freezing strategy: 'none', 'input_output'",
    type=str,
    default='none',
    choices=['none', 'early_1', 'early_2', 'early_3', 'early_4', 'late_1', 'late_2', 'late_3', 'late_4', 'input_output', 
         'resnet_1', 'resnet_2', 'resnet_3', 'resnet_12', 'resnet_23', 'resnet_123'],
    nargs='?')

args, unknown = parser.parse_known_args()
cobaya_yaml   = args.cobaya_yaml
probe         = args.probe
n_epochs      = args.n_epochs
batch_size    = args.batch_size
learning_rate = args.learning_rate
weight_decay  = args.weight_decay
save_losses   = args.save_losses
# ======== Additions by Béla =======  
save_testing_metrics = args.save_testing_metrics
squeeze_factor = args.squeeze_factor
# === TRANSFER LEARNING VARIABLES ===
transfer_learning = args.transfer_learning
pretrained_model = args.pretrained_model
freeze_strategy = args.freeze_strategy

#===================================================================================================
# covariance matrix read from file specified in the .dataset file specified in YAML
    
def get_cov(train_yaml):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'cov_file'):
            cov_file = split[2]

    full_cov = np.loadtxt(path+'/'+cov_file)
    cov_scenario = full_cov.shape[1]
    size = int(np.max(full_cov[:])+1)

    cov = np.zeros((size,size))

    for line in full_cov:
        i = int(line[0])
        j = int(line[1])

        if(cov_scenario == 3):
            cov_ij = line[2]
        elif(cov_scenario == 4):
            cov_ij = line[2]+line[3]
        elif(cov_scenario == 10):
            cov_ij = line[8]+line[9]

        cov[i,j] = cov_ij
        cov[j,i] = cov_ij

    cov = cov / squeeze_factor
    return cov

#===================================================================================================
# === TRANSFER LEARNING: Layer freezing function ===

def freeze_layers(model, freeze_strategy, transfer_learning=True):
    """
    Freeze layers based on transfer learning strategy.
    
    ResMLP structure:
    - model.0: Input layer (Linear: 12 -> 256)
    - model.1: ResBlock 1 
    - model.2: ResBlock 2
    - model.3: ResBlock 3  
    - model.4: Output layer (Linear: 256 -> 780)
    - model.5: Affine layer (gain, bias) (dont freeze!)
    """
    
    if not transfer_learning:
        return 0, sum(p.numel() for p in model.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = 0
    
    if freeze_strategy == 'none':
        print("TRANSFER LEARNING: No layers frozen - full fine-tuning")
        return frozen_params, total_params
    
    elif freeze_strategy == 'early_1':
        # Early Freezing 1: model.0 only (0.5% frozen)
        for param in model.model[0].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Early Freezing 1 - model.0 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: ResBlocks 1,2,3 + Output + Affine")
    
    elif freeze_strategy == 'early_2':
        # Early Freezing 2: model.0 + model.1 (22.6% frozen)
        for i in [0, 1]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Early Freezing 2 - model.0+1 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: ResBlocks 2,3 + Output + Affine")
    
    elif freeze_strategy == 'early_3':
        # Early Freezing 3: model.0 + model.1 + model.2 (44.6% frozen)
        for i in [0, 1, 2]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Early Freezing 3 - model.0+1+2 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: ResBlock 3 + Output + Affine")
    
    elif freeze_strategy == 'early_4':
        # Early Freezing 4: model.0 + model.1 + model.2 + model.3 (66.6% frozen)
        for i in [0, 1, 2, 3]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Early Freezing 4 - model.0+1+2+3 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Output + Affine only")
    
    elif freeze_strategy == 'late_1':
        # Late Freezing 1: model.4 only (33.3% frozen)
        for param in model.model[4].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Late Freezing 1 - model.4 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlocks 1,2,3 + Affine")
    
    elif freeze_strategy == 'late_2':
        # Late Freezing 2: model.4 + model.3 (55.3% frozen)
        for i in [4, 3]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Late Freezing 2 - model.4+3 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlocks 1,2 + Affine")
    
    elif freeze_strategy == 'late_3':
        # Late Freezing 3: model.4 + model.3 + model.2 (77.4% frozen)
        for i in [4, 3, 2]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Late Freezing 3 - model.4+3+2 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlock 1 + Affine")
    
    elif freeze_strategy == 'late_4':
        # Late Freezing 4: model.4 + model.3 + model.2 + model.1 (99.4% frozen)
        for i in [4, 3, 2, 1]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: Late Freezing 4 - model.4+3+2+1 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + Affine only")
    
    elif freeze_strategy == 'input_output':
        # Frozen Input and Output: model.0 + model.4 (33.8% frozen)
        for i in [0, 4]: 
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
    
        print(f"TRANSFER LEARNING: Input+Output Freezing - model.0+4 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: ResBlocks 1,2,3 + Affine")
    
    elif freeze_strategy == 'resnet_1':
        # Freeze only ResNet block 1
        for param in model.model[1].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: ResNet Block 1 Frozen - model.1 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlocks 2,3 + Output + Affine")
    
    elif freeze_strategy == 'resnet_12':
        # Freeze ResNet blocks 1+2
        for i in [1, 2]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: ResNet Blocks 1+2 Frozen - model.1+2 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlock 3 + Output + Affine")
    
    elif freeze_strategy == 'resnet_123':
        # Freeze ResNet blocks 1+2+3
        for i in [1, 2, 3]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: ResNet Blocks 1+2+3 Frozen - model.1+2+3 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + Output + Affine")
    
    elif freeze_strategy == 'resnet_2':
        # Freeze only ResNet block 2
        for param in model.model[2].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: ResNet Block 2 Frozen - model.2 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlocks 1,3 + Output + Affine")
    
    elif freeze_strategy == 'resnet_23':
        # Freeze ResNet blocks 2+3
        for i in [2, 3]:
            for param in model.model[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: ResNet Blocks 2+3 Frozen - model.2+3 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlock 1 + Output + Affine")
    
    elif freeze_strategy == 'resnet_3':
        # Freeze only ResNet block 3
        for param in model.model[3].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"TRANSFER LEARNING: ResNet Block 3 Frozen - model.3 ({frozen_params}/{total_params} = {100*frozen_params/total_params:.1f}%)")
        print("TRANSFER LEARNING: Trainable layers: Input + ResBlocks 1,2 + Output + Affine")
    
    else:
        raise ValueError(f"Unknown freeze_strategy: {freeze_strategy}")
    
    return frozen_params, total_params


#===================================================================================================
# a progress bar to display while training. I find tqdm to be a little strange looking.

def progress_bar(train_loss, valid_loss, start_time, epoch, total_epochs, optim):
    ''' 
    simple progress bar for training the emulator
    '''

    elapsed_time = int((datetime.now() - start_time).total_seconds())
    lr = optim.param_groups[0]['lr']
    epoch=epoch+1

    width = 20
    factor = int( width * (epoch/total_epochs) )
    bar = '['
    for i in range(width):
        if i < factor:
            bar += '#'
        else:
            bar += ' '
    bar += ']'

    remaining_time = int((elapsed_time / (epoch)) * (total_epochs - (epoch)))

    print('\r' + bar + ' ' +                                \
          f'Epoch {epoch:3d}/{total_epochs:3d} | ' +        \
          f'loss={train_loss:1.3e}({valid_loss:1.3e}) | ' + \
          f'lr={lr:1.2e} | ' +                              \
          f'time elapsed={elapsed_time:7d} s; time remaining={remaining_time:7d} s',end='')

#===================================================================================================
# training routine

def train_emulator(train_yaml, probe,
            n_epochs=250, batch_size=32, learning_rate=1e-3, weight_decay=0, 
            save_losses=False, save_testing_metrics=False, squeeze_factor=1.0,
            transfer_learning=False, pretrained_model=None, freeze_strategy='none'):
    '''
    routine to train an emulator. 

    string  train_yaml: the training YAML file. See 'projects/lsst_y1/EXAMPLE_TRAIN.yaml'
    string  probe: the probe name (e.g., 'cosmic_shear', 'galaxy_clustering', 'galaxy_galaxy_lensing')
    int     n_epochs: number of training epochs (default=250)
    int     batch_size: batch size for training (default=32)
    float   learning_rate: learning rate for ADAM optimizer (default=1e-3)
    float   weight_decay: L2 regularization weight decay (default=0)
    boolean save_losses: save training and validation loss to 'losses.txt' (default=False)
    boolean save_testing_metrics: save testing metrics to 'testing_metrics.txt' (default=False)
    float   squeeze_factor: factor to divide covariance matrix by (default=1.0, no squeezing)
    boolean transfer_learning: use transfer learning from pretrained model (default=False)
    string  pretrained_model: path to pretrained model file (required if transfer_learning=True)
    string  freeze_strategy: layer freezing strategy - 'none', 'early_1', 'early_2', 'early_3', 'early_4', 
                           'late_1', 'late_2', 'late_3', 'late_4', 'input_output' (default='none')
    '''
    print('')
    print('Probe =', probe)

    # TODO: get indices from cosmolike
    if probe=='galaxy_galaxy_lensing':
        start = 780
        stop = 780+650

    elif probe=='cosmic_shear':
        start = 0
        stop = 780

    elif probe=='galaxy_clustering':
        start=780+650
        stop = 1560

    else:
        raise NotImplementedError

    # open the config file and get the arguments we want.
    with open(train_yaml,'r') as stream:
        args = yaml.safe_load(stream)

    # get training and validation_data
    if args['train_args']['training_data_path'][0] == '/':
        PATH = args['train_args']['training_data_path']
    else:
        PATH = os.environ.get("ROOTDIR") + '/' + args['train_args']['training_data_path']

    # get saving filenames      
    model_filename = args['train_args'][probe]['extra_args']['file'][0]
    extra_filename = args['train_args'][probe]['extra_args']['extra'][0]

    print('Model will be saved to:',model_filename,'and',extra_filename)
    print('')

    covmat_file      = PATH + args['train_args']['data_covmat_file']

    train_datavectors_file = PATH + args['train_args']['train_datavectors_file']
    train_parameters_file  = PATH + args['train_args']['train_parameters_file']

    valid_datavectors_file = PATH + args['train_args']['valid_datavectors_file']
    valid_parameters_file  = PATH + args['train_args']['valid_parameters_file']

    test_datavectors_file = PATH + args['train_args']['test_datavectors_file']
    test_parameters_file  = PATH + args['train_args']['test_parameters_file']

    print('Loading training data from:')
    print(train_datavectors_file)
    print(train_parameters_file)
    print('Loading validation data from:')
    print(valid_datavectors_file)
    print(valid_parameters_file)
    print('')

    # get the parameters
    sampled_params = args['train_args'][probe]['extra_args']['ord'][0]
    sampling_dim = len(sampled_params)

    print('Parameters are:')
    print(sampled_params)
    print('')

    # get device
    device = args['train_args'][probe]['extra_args']['device']

    # get model
    model_info = args['train_args'][probe]['extra_args']['extrapar'][0]

    if( 'TRF' == model_info['MLA'] ):
        model = ResTRF(sampling_dim, 
            model_info['OUTPUT_DIM'], 
            model_info['INT_DIM_RES'], 
            model_info['INT_DIM_TRF'],
            model_info['NC_TRF'])
    elif( 'MLP' == model_info['MLA'] ):
        model = ResMLP(sampling_dim,
            model_info['OUTPUT_DIM'],
            model_info['INT_DIM_RES'])
    else:
        raise NotImplementedError
    
    # === TRANSFER LEARNING: Load pretrained model if specified ===
    if transfer_learning:
        if pretrained_model is None:
            raise ValueError("transfer_learning=True but no pretrained_model specified!")
        
        print(f'\nTRANSFER LEARNING: Loading pretrained model from {pretrained_model}')
        
        # Load pretrained weights
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        model.load_state_dict(pretrained_state)
        
        # Load pretrained preprocessing parameters
        pretrained_h5 = pretrained_model + '.h5'
        with h5.File(pretrained_h5, 'r') as f:
            pretrained_samples_mean = torch.tensor(f['sample_mean'][:])
            pretrained_samples_std = torch.tensor(f['sample_std'][:])
            pretrained_dv_fid = torch.tensor(f['dv_fid'][:])
            pretrained_dv_evals = torch.tensor(f['dv_evals'][:])
            pretrained_dv_evecs = torch.tensor(f['dv_evecs'][:])
        
        print(f'TRANSFER LEARNING: Loaded pretrained preprocessing parameters')
        print(f'TRANSFER LEARNING: Using freeze strategy: {freeze_strategy}')
        
        # Apply layer freezing
        frozen_params, total_params = freeze_layers(model, freeze_strategy, transfer_learning=True)
    else:
        # Normal training - no pretrained parameters
        pretrained_samples_mean = None
        pretrained_samples_std = None
        pretrained_dv_fid = None
        pretrained_dv_evals = None
        pretrained_dv_evecs = None
        
        frozen_params, total_params = freeze_layers(model, 'none', transfer_learning=False)


    # load and preprocess the data
    print('Processing the data. May take some time...')

    # read the header from the train_parameters:
    f_train = open(train_parameters_file)
    train_params = np.array(f_train.readline().split(' ')[1:])
    train_params[-1] = train_params[-1][:-1] # because the last param has a \n

    f_valid = open(valid_parameters_file)
    valid_params = np.array(f_valid.readline().split(' ')[1:])
    valid_params[-1] = valid_params[-1][:-1] # because the last param has a \n

    f_test = open(test_parameters_file)
    test_params = np.array(f_test.readline().split(' ')[1:])
    test_params[-1] = test_params[-1][:-1] # because the last param has a \n

    train_idxs = []
    valid_idxs = []
    test_idxs  = []
    for p in sampled_params:
        train_idxs.append(np.where(train_params==p)[0][0])
        valid_idxs.append(np.where(valid_params==p)[0][0])
        test_idxs.append(np.where(test_params==p)[0][0])

    # load the data of the given train_prefix and valid_prefix. Leave on cpu to save vram!
    x_train = torch.as_tensor(np.loadtxt(train_parameters_file)[:,train_idxs],dtype=torch.float64)
    y_train = torch.as_tensor(np.load(train_datavectors_file)[:,start:stop],dtype=torch.float64)

    x_valid = torch.as_tensor(np.loadtxt(valid_parameters_file)[:,valid_idxs],dtype=torch.float64)
    y_valid = torch.as_tensor(np.load(valid_datavectors_file)[:,start:stop],dtype=torch.float64)

    x_test = torch.as_tensor(np.loadtxt(test_parameters_file)[:,test_idxs],dtype=torch.float64)
    y_test = torch.as_tensor(np.load(test_datavectors_file)[:,start:stop],dtype=torch.float64)

    # convert data
    covmat = torch.as_tensor(get_cov(train_yaml)[start:stop,start:stop],dtype=torch.float64)
    dv_fid  = torch.as_tensor(torch.mean(y_train[start:stop],axis=0),dtype=torch.float64)

    # === TRANSFER LEARNING: Choose preprocessing strategy ===
    if transfer_learning:
        # Use pretrained preprocessing for consistency
        samples_mean = pretrained_samples_mean
        samples_std = pretrained_samples_std
        print('TRANSFER LEARNING: Using pretrained preprocessing parameters')
    else:
        # Normal training - compute new preprocessing parameters
        samples_mean = torch.Tensor(x_train.mean(axis=0, keepdims=True))
        samples_std  = torch.Tensor(x_train.std(axis=0, keepdims=True))
        print('NORMAL TRAINING: Computing new preprocessing parameters')

    x_train = torch.div( (x_train - samples_mean), 5*samples_std)
    x_valid = torch.div( (x_valid - samples_mean), 5*samples_std)
    x_test  = torch.div( (x_test  - samples_mean), 5*samples_std)

    # diagonalize the training datavectors
    dv_evals, dv_evecs = torch.linalg.eigh(covmat)
    inv_covmat = torch.diag(1/dv_evals).type(torch.float32).to(device)

    y_train = torch.div( (y_train - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))
    y_valid = torch.div( (y_valid - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))
    y_test  = torch.div( (y_test  - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))

    # convert to float32
    x_train = torch.as_tensor(x_train,dtype=torch.float32)
    y_train = torch.as_tensor(y_train,dtype=torch.float32)
    x_valid = torch.as_tensor(x_valid,dtype=torch.float32)
    y_valid = torch.as_tensor(y_valid,dtype=torch.float32)
    x_test  = torch.as_tensor(x_test, dtype=torch.float32)
    y_test  = torch.as_tensor(y_test, dtype=torch.float32)

    # === TRANSFER LEARNING: Setup optimizer for trainable parameters only ===
    if transfer_learning:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        print(f'TRANSFER LEARNING: Optimizer using {len(trainable_params)} parameter groups')
    else:
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print('NORMAL TRAINING: Optimizer using all model parameters')
    
    # scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',patience=15,factor=0.1)
     
    # load the data into loaders
    model.to(device)

    generator = torch.Generator(device=device)
    trainset    = torch.utils.data.TensorDataset(x_train, y_train)
    validset    = torch.utils.data.TensorDataset(x_valid, y_valid)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)

    # begin training
    print('Begin training...',end='')
    train_start_time = datetime.now()

    losses_train = []
    losses_valid = []
    loss = 100.
    #initialize testing metrics lists
    test_mean_chi2 = []
    test_median_chi2 = []
    test_frac_gt_0p2 = []
    test_frac_gt_1 = []
    # ======= New additions from Béla =======
    test_frac_lt_0p2 = []      # fraction with chi^2 < 0.2
    test_criterion_met = []    # boolean if criterion is satisfied

    for e in range(n_epochs):
        model.train()

        # training loss
        losses = []
        for i, data in enumerate(trainloader):    
            X       = data[0].to(device)
            Y_batch = data[1].to(device)
            Y_pred  = model(X)

            # PCA part
            diff = Y_batch - Y_pred
            chi2 = torch.diag(diff @ torch.t(diff))

            # loss = torch.mean(chi2)                      # ordinary chi2
            loss = torch.mean((1+2*chi2)**(1/2))-1       # hyperbola
            # loss = torch.mean(torch.mean(chi2**(1/2)))   # sqrt(chi2)

            losses.append(loss.cpu().detach().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

        losses_train.append(np.mean(losses))

        ###validation loss
        losses=[]
        with torch.no_grad():
            model.eval()
            losses = []
            for i, data in enumerate(validloader):  
                X_v       = data[0].to(device)
                Y_v_batch = data[1].to(device)
                Y_v_pred = model(X_v)

                diff_v = Y_v_batch - Y_v_pred
                chi2_v = torch.diag(diff_v @ torch.t(diff_v))

                # loss_vali = torch.mean(chi2_v)                      # ordinary chi2
                loss_vali = torch.mean((1+2*chi2_v)**(1/2))-1       # hyperbola
                # loss_vali = torch.mean(torch.mean(chi2_v**(1/2)))   # sqrt(chi2)

                losses.append(float(loss_vali.cpu().detach().numpy()))

            losses_valid.append(np.mean(losses))

            scheduler.step(losses_valid[e])
            optim.zero_grad()

        ### Testing metrics at each epoch
        with torch.no_grad():
            Y_t = model(x_test.to(device))
            delta_chi2 = torch.diag((y_test.to(device) - Y_t) @ torch.t(y_test.to(device) - Y_t))
            
            # Compute metrics
            mean_chi2 = torch.mean(delta_chi2).cpu().detach().numpy()
            median_chi2 = torch.median(delta_chi2).cpu().detach().numpy()
            
            # Count fractions
            n_total = len(delta_chi2)
            n_gt_0p2 = torch.sum((delta_chi2 > 0.2).float()).cpu().detach().numpy()
            n_gt_1 = torch.sum((delta_chi2 >= 1).float()).cpu().detach().numpy()
            # ====== New addition: > to < ===== 
            n_lt_0p2 = torch.sum((delta_chi2 < 0.2).float()).cpu().detach().numpy()
            
            # ====== New computing fractions =-=====
            frac_gt_0p2 = n_gt_0p2 / n_total   # fraction with chi^2 > 0.2
            frac_gt_1 = n_gt_1 / n_total       # fraction greater than 1
            frac_lt_0p2 = n_lt_0p2 / n_total   # fraction with chi^2 < 0.2
            criterion_met = frac_lt_0p2 > 0.1  # fractional criterion

            # Append to lists
            test_mean_chi2.append(mean_chi2)
            test_median_chi2.append(median_chi2)
            test_frac_gt_0p2.append(frac_gt_0p2)   # same as before, but defining fraction outside of testing
            test_frac_gt_1.append(frac_gt_1)       # same as before, but defining fraction outside of testing
            test_frac_lt_0p2.append(frac_lt_0p2)   # New
            test_criterion_met.append(float(criterion_met)) # New


        progress_bar(losses_train[-1],losses_valid[-1],train_start_time, e, n_epochs, optim)
    
    if ( save_losses ):
        np.savetxt("losses.txt", np.array([losses_train,losses_valid],dtype=np.float64))

    if ( save_testing_metrics ):
        np.savetxt("testing_metrics.txt", np.array([
            test_mean_chi2,           # mean chi2
            test_median_chi2,         # median chi2  
            test_frac_gt_0p2,         # fraction > 0.2
            test_frac_gt_1,           # fraction > 1.0
            test_frac_lt_0p2,         # NEW fraction < 0.2 
            test_criterion_met        # NEW criterion met (1 if True, 0 if False)
        ], dtype=np.float64))

    # save the model
    torch.save(model.state_dict(), model_filename)
    with h5.File(extra_filename, 'w') as f:
        f['sample_mean']   = samples_mean
        f['sample_std']    = samples_std
        f['dv_fid']        = dv_fid
        f['dv_evals']      = dv_evals
        f['dv_evecs']      = dv_evecs
        f['train_params']  = sampled_params


    # now lets test the model
    print('')
    print('Testing the model...')
    print('')
    # Reminder:
    # C = UDU^{-1}
    # dv_norm = D^{-1/2} U^{-1} dv
    # where D is diagonal and U is orthogonal, so
    # dv_norm.T = dv.T U D^{-1/2}
    # 
    # chi2 = dv.T @ C @ dv
    #      = dv.T @ UDU^{-1} @ dv
    #      = dv.T U @ D^{-1/2} D^{-1/2} @ U^{-1} dv
    #      = (dv.T U D^{-1/2}) @ (D^{-1/2} U^{-1} dv)
    #      = dv_norm.T @ dv_norm
    # and dv_norm is just the model output!

    Y_t = model(x_test.to(device))
    delta_chi2 = torch.diag((y_test.to(device) - Y_t) @ torch.t(y_test.to(device) - Y_t))

    chi2_g_1  = 0
    chi2_g_p2 = 0

    for c in delta_chi2:
        if( c>0.2 and c<1 ):
            chi2_g_p2 += 1
        elif( c>=1 ):
            chi2_g_1 += 1

    print('Testing results.')
    print('Mean   Delta Chi2 = {:1.3e}'.format(torch.mean(delta_chi2).cpu().detach().numpy()))
    print('Median Delta Chi2 = {:1.3e}'.format(torch.median(delta_chi2).cpu().detach().numpy()))
    print('N points with Chi2 > 1  :', chi2_g_1)
    print('N points with Chi2 > 0.2:', chi2_g_p2)
    # === New additions by Béla ===
    print("N points with Chi2 < 0.2: {}".format(n_lt_0p2)) 
    print("Fraction with Chi2 < 0.2: {:.3f}".format(frac_lt_0p2))
    print("Fractional criterion (>0.1): {} (Target: True)".format(criterion_met))


    # Done :)
    print('\nDone!')

    return

if __name__ == "__main__":
    train_emulator(cobaya_yaml, probe, 
        n_epochs, batch_size, learning_rate, weight_decay, 
        save_losses, save_testing_metrics, squeeze_factor,
        transfer_learning, pretrained_model, freeze_strategy)
