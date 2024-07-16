from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# import network
from model.AerialNet import dinov2_net_two_stream as net
from model.loss import TransformedGridLoss
from data.download import download_train
from data.synth_dataset import SynthDataset
from geotnf.transformation import SynthPairTnf
from image.normalization import NormalizeImageDict
from util.train_test_fn import train, test
from util.torch_util import save_checkpoint, str_to_bool, print_info


# torch.cuda.set_device(1) # Using second GPU

if __name__ == '__main__':
    print_info('[Deep Aerial Matching] training script',['green','bold'])

    # Argument parsing
    parser = argparse.ArgumentParser(description='Deep Aerial Matching PyTorch Implementation')
    # Paths
    parser.add_argument('--training-dataset', type=str, default='Cities', help='dataset to use for training')
    parser.add_argument('--training-tnf-csv', type=str, default='./training_data/Cities/', help='path to training transformation csv folder')
    parser.add_argument('--training-image-path', type=str, default='./', help='path to folder containing training images')
    parser.add_argument('--trained-models-dir', type=str, default='trained_models', help='path to trained models folder')
    parser.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='trained model filename')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=12, help='training batch size')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
    # Model parameters
    parser.add_argument('--geometric-model', type=str, default='affine', help='geometric model to be regressed at output: affine parameters (6 degrees of freedom)')
    parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')
    parser.add_argument('--dinov2-model_name', type=str, default='dinov2_vitb14', help='Feature extraction architecture: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14')
    parser.add_argument('--train-fe', type=str_to_bool, nargs='?', const=True, default=True, help='True: train feature extraction or False: freeze feature extraction')
    # Synthetic dataset parameters
    parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False, help='sample random transformations')
    # Reload model parameter
    parser.add_argument('--load-model', type=bool, default=False, help='loading the trained model checkpoint')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    # Seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)


    # Download dataset if needed and set paths
    if args.training_dataset == 'GoogleEarth':
        if args.training_image_path == '':
            args.training_image_path = 'datasets/'
        # Download dataset
        download_train(args.training_image_path+args.training_dataset)
        if args.training_tnf_csv == '' and args.geometric_model=='affine':
            args.training_tnf_csv = 'training_data/GoogleEarth-aff'

    # CNN model and loss
    print('Creating DINOv2 model...')
    model = net(geometric_model=args.geometric_model, dinov2_model_name=args.dinov2_model_name,train_fe=args.train_fe,num_trainable_blocks=4)

    if args.use_mse_loss:
        print('Using MSE loss...')
        loss = nn.MSELoss()
    else:
        print('Using grid loss...')
        loss = TransformedGridLoss(use_cuda=use_cuda,geometric_model=args.geometric_model)


    # Dataset and dataloader
    dataset_train = SynthDataset(geometric_model=args.geometric_model,
                           csv_file=os.path.join(args.training_tnf_csv,'train_pairs.csv'),
                           training_image_path=args.training_image_path,
                           transform=NormalizeImageDict(['src_image','trg_image','trg_image_jit']),
                           random_sample=args.random_sample)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    dataset_test = SynthDataset(geometric_model=args.geometric_model,
                                csv_file=os.path.join(args.training_tnf_csv,'val_pairs.csv'),
                                training_image_path=args.training_image_path,
                                transform=NormalizeImageDict(['src_image','trg_image','trg_image_jit']),
                                random_sample=args.random_sample)

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4)

    pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,use_cuda=use_cuda)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam([{'params':model.FeatureExtraction.parameters()},{'params':model.FeatureRegression.parameters(),'lr':1e-3}], lr=args.lr)


    # The number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Train
    best_test_loss = float("inf")

    print('Starting training...\n')
    print_info("# ===================================== #\n"
               "\t\t\t...Train config...\n"
               "\t\t\t------------------\n"
               "\t\t CNN model: "+args.dinov2_model_name+"\n"
               "\t\t Geometric model: "+args.geometric_model+"\n"
               "\t\t Dataset: "+args.training_dataset+"\n"
               "\t\t # of train data: "+str(len(dataset_train))+"\n\n"
               "\t\t Learning rate: "+str(args.lr)+"\n"
               "\t\t Batch size: "+str(args.batch_size)+"\n"
               "\t\t Maximum epoch: "+str(args.num_epochs)+"\n"
               "\t\t Reload checkpoint: "+str(args.load_model)+"\n\n"
               "\t\t # of parameters: "+str(total_params)+"\n"
               "# ===================================== #\n",['yellow','bold'])

    if args.load_model:
        load_dir = 'trained_models/checkpoint_seresnext101.pth.tar'
        checkpoint = torch.load(load_dir, map_location=lambda storage, loc: storage)  # Load trained model

        # Load parameters of FeatureExtraction
        for name, param in model.FeatureExtraction.state_dict().items():
            model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])
        # Load parameters of FeatureRegression (Affine)
        for name, param in model.FeatureRegression.state_dict().items():
            model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
        print("Reloading from--[%s]" % load_dir)

    for epoch in range(1, args.num_epochs+1):
        # Call train, test function
        train_loss = train(epoch,model,loss,optimizer,dataloader_train,pair_generation_tnf,log_interval=100)
        # test_loss = test(model,loss,dataloader_test,pair_generation_tnf)

        if args.use_mse_loss:
            checkpoint_name = os.path.join(args.trained_models_dir,args.geometric_model+'_mse_loss_'+args.feature_extraction_cnn+'_'+args.training_dataset+'_epoch_'+str(epoch)+'.pth.tar')
        else:
            checkpoint_name = os.path.join(args.trained_models_dir,args.geometric_model+'_grid_loss_'+args.feature_extraction_cnn+'_'+args.training_dataset+'_epoch_'+str(epoch)+'.pth.tar')
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        },checkpoint_name)

    print('Done!')