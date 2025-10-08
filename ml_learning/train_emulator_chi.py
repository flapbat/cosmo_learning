import torch
import numpy as np
import os
import sys
from datetime import datetime
from emulator import ResTRF, ResMLP
import yaml
import h5py as h5
import yaml
import numpy as np
import os
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

args, unknown = parser.parse_known_args()
cobaya_yaml   = args.cobaya_yaml
probe         = args.probe
n_epochs      = args.n_epochs
batch_size    = args.batch_size
learning_rate = args.learning_rate
weight_decay  = args.weight_decay
save_losses   = args.save_losses

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

    return cov

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

def chichi(m, x_test, y_test):
    Y_t = m(x_test.to(device))
    delta_chi2 = torch.diag((y_test.to(device) - Y_t) @ torch.t(y_test.to(device) - Y_t))

    chi2_g_1  = 0
    chi2_g_p2 = 0

    for c in delta_chi2:
        if( c>0.2 and c<1 ):
            chi2_g_p2 += 1
        elif( c>=1 ):
            chi2_g_1 += 1

    meanchi2=torch.mean(delta_chi2).cpu().detach().numpy()
    medianchi2=torch.median(delta_chi2).cpu().detach().numpy()
    np.savetxt("chi2.txt", np.array([meanchi2,medianchi2,chi2_g_p2, chi2_g_1],dtype=np.float64))




def train_emulator(train_yaml, probe,
            n_epochs=250, batch_size=32, learning_rate=1e-3, weight_decay=0, 
            save_losses=False):
    '''
    routine to train an emulator. 

    string  train_yaml: the training YAML file. See 'projects/lsst_y1/EXAMPLE_TRAIN.yaml'
    boolean save_losses: save the training and validation loss to a text file 'losses_*.txt'
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

    # normalize the input parameters
    samples_mean = torch.Tensor(x_train.mean(axis=0, keepdims=True))
    samples_std  = torch.Tensor(x_train.std(axis=0, keepdims=True))

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

    # setup ADAM optimizer and reduce_lr scheduler
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

        progress_bar(losses_train[-1],losses_valid[-1],train_start_time, e, n_epochs, optim)
        chichi(model, x_test, y_test)
    
    if ( save_losses ):
        np.savetxt("losses.txt", np.array([losses_train,losses_valid],dtype=np.float64))

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

    # Done :)
    print('\nDone!')

    return

if __name__ == "__main__":
    train_emulator(cobaya_yaml, probe, 
        n_epochs, batch_size, learning_rate, weight_decay, 
        save_losses)
