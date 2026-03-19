import argparse
import pprint


def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train deep implicit 3D geometry representations.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--exp_name', type=str,
                              help='Experiment name.')
    global_group.add_argument('--validator', type=str, default=None,
                              help='Run validation.')
    global_group.add_argument('--valid-only', action='store_true',
                              help='Run validation (and do not run training).')
    global_group.add_argument('--valid-every', type=int, default=1,
                             help='Frequency of running validation.')
    global_group.add_argument('--debug', action='store_true',
                              help='Utility argument for debug output and viz.')
    global_group.add_argument('--seed', type=int,
                              help='NumPy random seed.')
    global_group.add_argument('--ngc', action='store_true',
                              help='Use NGC arguments.')

    # Architecture for network
    net_group = parser.add_argument_group('net')
    net_group.add_argument('--net', type = str, default = 'OctreeSDF', 
                          help='The network architecture to be used.')
    net_group.add_argument('--num_layers', type = int, default = 1,
                           help='The number of layers in the decoder.')
    net_group.add_argument('--hidden_dim', type=int, default=128)
    net_group.add_argument('--code_res', type=int, default=9)
    net_group.add_argument('--emb_deg', type=int, default=256)
    net_group.add_argument('--activation', type=str, default='sp',choices=['relu','sp','sine'],
                           help = 'The activation function to be used.')
    net_group.add_argument('--pretrained', type=str, default=None)
    net_group.add_argument('--discard_latent', action='store_false',help='Whether to discard the latent codes.')
    net_group.add_argument('--fix_decoder', action='store_true',help='Whether to fix the decoder.')


    # Arguments for dataset
    data_group = parser.add_argument_group('dataset')

    # Octree
    data_group.add_argument('--start_depth', type=int,default = 4,
                            help='The starting depth of the octree')
    data_group.add_argument('--end_depth', type=int,default = 4,
                            help='The end depth of the octree')

    # Mesh Dataset
    data_group.add_argument('--dataset_path', type=str,
                            help='Path of dataset')
    data_group.add_argument('--dataset_prefix', type=str, default='data/train',
                            help='Prefix of the dataset path')
    data_group.add_argument('--dataset_save', type=str, default='data/samples',
                            help='Prefix for saving the presampled points.')
    data_group.add_argument('--shape_list', nargs='+', type=str, default = [
         '00000002','00000003','00000004','00000008'])
    
    data_group.add_argument('--resolution_power',type=int, default=4,
                            help='the resolution of the patches')
    data_group.add_argument('--merge_resolution_power',type=int, default=4,
                            help='the resolution of the merge process')
    data_group.add_argument('--total_surf_num', type = int, default = 5000000)
    data_group.add_argument('--batch_spatial_num', type = int, default = 5000)
    data_group.add_argument('--batch_merge_num', type = int, default = 5000)
    data_group.add_argument('--batch_surf_num', type = int, default = 5000)
    
    #data_group.add_argument('--batch_bdr_num', type = int, default = 1000)
    data_group.add_argument('--batch_bdr_num', type = int, default = 0)
    data_group.add_argument('--batch_jitter_surf_num', type = int, default = 5000)
    data_group.add_argument('--box_uniform_num', type = int, default = 1000)
    #data_group.add_argument('--box_uniform_num', type = int, default = 200)
    data_group.add_argument('--box_surf_num', type = int, default = 1000)
    data_group.add_argument('--box_bdr_num', type = int, default = 1000)
    data_group.add_argument('--jitter_delta_scale', type = float, default = 0.1)
    data_group.add_argument('--merge_extend_length', type = float, default = 0.005)
    data_group.add_argument('--box_equal_surf', action='store_false', 
                            help = 'If we want to equally sample surface points in the box')

    
    # Arguments for optimizer
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--decoder_lr', type = float, default = 0.001, 
                             help = 'Learning rate for decoder.')
    optim_group.add_argument('--latent_lr', type = float, default = 0.001, 
                             help = 'Learning rate for latent codes.')
    optim_group.add_argument('--lr_scheduler', type = str, default = 'multi_step', choices = ['multi_step', 'exp'],
                             help = 'Can choose from exp and multi_step.')
    optim_group.add_argument('--lr_exp_gamma', type = float, default = 0.999, 
                             help = 'Decay gamma for the exp lr scheduler.')
    optim_group.add_argument('--lr_multi_gamma', type = float, default = 0.3, 
                             help = 'Decay gamma for the multi step lr scheduler.')
    optim_group.add_argument('--mile_stone', type = int, nargs = 2, default = [270, 285],
                             help = 'Mile stones for lr decay.')
 
    # Arguments for training
    train_group = parser.add_argument_group('trainer')             
    train_group.add_argument('--epochs', type = int, default = 300,                                          
                             help='Number of epochs to run the training.')
    train_group.add_argument('--retreat_bdr_nrm_epoch',type = int,default = 10)         
    train_group.add_argument('--print_loss_every', type = int, default = 10,
                             help='Print information every N epochs')
    train_group.add_argument('--model_path', type = str, default = '_results/models', 
                             help='Path to save the trained models.')
    train_group.add_argument('--use_shape_space', action = "store_true", 
                             help='To use the shape space or not. If use, the decoder would be fixed.')
    train_group.add_argument('--after_merge_supervision', action = "store_true", 
                             help='Use after merge supervision for smooth connections.')
    train_group.add_argument('--latent_path', type = str, default = '_results/shape_space', 
                             help='Path to save the latent space related models.')
    train_group.add_argument('--save-as-new', action = 'store_true', 
                             help='Save the model at every epoch (no overwrite).')
    train_group.add_argument('--save-every', type = int, default = 50, 
                             help='Save the model at every N epoch.')
    train_group.add_argument('--save-all', action = 'store_true', 
                             help='Save the entire model')
    train_group.add_argument('--latent', action = 'store_true', 
                             help='Train latent space.')
    train_group.add_argument('--return-lst', action = 'store_true', 
                             help='Returns a list of predictions (optimization).')
    train_group.add_argument('--latent-dim', type = int, default = 128, 
                             help='Latent vector dimension.')
    train_group.add_argument('--logs', type = str, default = '_results/logs/runs/',
                             help='Log file directory for checkpoints.')

    # Arguments for loss
    loss_group = parser.add_argument_group('loss')

    loss_group.add_argument('--surf', type = float, default = 200)    
    loss_group.add_argument('--surf_normal', type = float, default = 50)
    loss_group.add_argument('--merge_surf', type = float, default = 400)
    loss_group.add_argument('--jitter_sdf', type = float, default = 50)
    loss_group.add_argument('--eikonal', type = float, default = 5)

    loss_group.add_argument('--top_surf', type = float, default = 0)
    loss_group.add_argument('--penalty', type = float, default = 0.0)

    loss_group.add_argument('--merge_smooth_surf', type = float, default = 0)
    loss_group.add_argument('--impossible_num', type = int, default = -100)
    loss_group.add_argument('--bdr', type = float, default = 0)
    loss_group.add_argument('--start_top_epoch', type = int, default = 150)
    loss_group.add_argument('--merge_err_threshold', type = float, default = 0.001)

    loss_group.add_argument('--margin_v', type = float, default = 0.0)
    loss_group.add_argument('--use_smooth', action = 'store_false')
    loss_group.add_argument('--add_box_extension', action = 'store_false',
                            help = 'Extend the box to get more samples, so that to protect the boundary of the box.')
    loss_group.add_argument('--add_box_boundary', action = 'store_true',
                            help = 'Add points on the boundary of the boxes to strengthen the learning of the boundary of the boxes.')
    loss_group.add_argument('--box_extension_scale', type=float, default = 1.2)
    loss_group.add_argument('--jitter_surface', action= 'store_false')
    loss_group.add_argument('--single_patch', action= 'store_true')
    loss_group.add_argument('--update', action= 'store_true',
                            help = 'The case for updating a shape.')
    loss_group.add_argument('--follow_shape_name', type=str, default = None)
    loss_group.add_argument('--open', action= 'store_true',
                            help = 'Whether it is open surface or not.')
    loss_group.add_argument('--train_shape_space', action= 'store_true',
                            help = 'Whether it is open surface or not.')
    loss_group.add_argument('--retreat_bdr_nrm', action= 'store_true',
                            help = 'For open boundaries, retreat the boundary normals supervision after certain iterations.')
    loss_group.add_argument('--later_flex_decoder', action= 'store_true')
    loss_group.add_argument('--flex_decoder_epochs', type=int, default = 200)
    loss_group.add_argument('--enhance_normal_epochs', type=int, default = 100)
    loss_group.add_argument('--later_enhance_normal', action= 'store_true',
                            help = 'Increase the normal supervision after certain iterations.')


    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str
