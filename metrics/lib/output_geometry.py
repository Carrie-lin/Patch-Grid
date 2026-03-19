#import open3d as o3d
import numpy as np
from skimage import measure
import trimesh
#import open3d.visualization.rendering as rendering
import os

from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter ## this is for lei's conda env
import torch
# from chamferdist import ChamferDistance
import argparse
import shutil
from torch import optim
import time
from pysdf import SDF

## aim: generate surface from the raw output of the network
# def my_marching_cube(model,epoch_num,real_points,res_dir,max_resolu):
def my_marching_cube(filename, model, max_resolu):

    
    low=-0.5
    high=0.5
    grid = np.linspace(low,high, max_resolu)
    net_input=[[[[x,y,z] for x in grid] for y in grid] for z in grid]
    
    net_input=torch.FloatTensor(net_input)

    net_input=net_input.reshape((max_resolu*max_resolu*max_resolu,3))

    max_len=net_input.shape[0]
    one_round_batchsize=100000

    with torch.no_grad():
        cur_input=net_input[0:one_round_batchsize].cuda()
        net_output=model(cur_input)
        net_output=net_output.cpu()
        
        ## TODO: you can compute the max value of i by the following
        ## ceil(max_len / one_round_batchsize)
        for i in range(1,1000000): 
            if (i+1)*one_round_batchsize>max_len:
                cur_input=net_input[i*one_round_batchsize:max_len]
                cur_input=cur_input.cuda()
                cur_output=(model(cur_input)).cpu()
                net_output=torch.cat((net_output,cur_output),axis=0)
                break
            else:
                cur_input=net_input[i*one_round_batchsize:(i+1)*one_round_batchsize]
                cur_input=cur_input.cuda()
                cur_output=(model(cur_input)).cpu()
                net_output=torch.cat((net_output,cur_output),axis=0)

    net_output=net_output.reshape(max_resolu,max_resolu,max_resolu)
    net_output=net_output.numpy()
    net_output=np.transpose(net_output,(2,1,0))

    ## TODO: is one_axis_points == max_resolu?
    ## Answer: Yes, one_axis_points is for computing the interval between 2 points
    one_axis_points=net_output.shape[0]-1
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume=net_output,level=0,spacing=(1/one_axis_points,1/one_axis_points,1/one_axis_points))
    
    ## let all the vertices start from [-0.5,-0.5,-0.5] 
    vertices = verts + np.array([-0.5, -0.5, -0.5])

    meshexport = trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=False)
    meshexport.export(filename)
    return meshexport

def gen_mc_mask(tri_mesh, grid_pnts, clamp_dist=0.1):
    # surf_pnts = surf_pnts.detach().cpu().numpy()
    # surf_tree = cKDTree(surf_pnts)
 
    dists = [] #np.zeros(grid_pnts.shape[0])
    sdf_func = SDF(tri_mesh.vertices, tri_mesh.faces,False)
    dists=abs(sdf_func(grid_pnts))
    #dists = np.concatenate(dists)
    mask = np.abs(dists) < clamp_dist
    return mask
    

def marching_cube(folder,device, model, max_resolu,one_round_batch):

    #x_low=-0.15#-1.0 #-0.8
    #x_high=0.15 #  1.0 #0 
    #y_low=0.6#-1.0 #-0.8
    #y_high=0.9 #  1.0 #0
    #z_low=0.6#-1.0
    #z_high=0.9 #  1.0

    x_low= -1#-1.0 #-0.8
    x_high= 1 #  1.0 #0 
    y_low= -1#-1.0 #-0.8
    y_high= 1 #  1.0 #0
    z_low= -1#-1.0
    z_high= 1 #  1.0

    x_grid = np.linspace(x_low,x_high, max_resolu)
    y_grid = np.linspace(y_low,y_high, max_resolu)
    z_grid = np.linspace(z_low,z_high, max_resolu)

    #print('1')

    length = x_high - x_low

    xg, yg, zg = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    #print('22')
    xg = xg.reshape(-1)
    yg = yg.reshape(-1)
    zg = zg.reshape(-1)
    #print('23')

    net_input = np.stack((zg, yg, xg), axis=-1)
    net_input=torch.FloatTensor(net_input)
    net_input=net_input.reshape((max_resolu*max_resolu*max_resolu,3))
    max_len=net_input.shape[0]
    batchsize=one_round_batch
    #print('3')

    with torch.no_grad():
        cur_input=net_input[0:batchsize].to(device)
        net_output=model(cur_input)
        net_output=net_output.cpu()
        #print("1111111")
        ## TODO: you can compute the max value of i by the following
        ## ceil(max_len / one_round_batchsize)
        for i in range(1,1000000): 
            #print(i)
            if (i+1)*batchsize>max_len:
                cur_input=net_input[i*batchsize:max_len]
                cur_input=cur_input.to(device)
                cur_output=(model(cur_input)).cpu()
                net_output=torch.cat((net_output,cur_output),axis=0)
                break
            else:
                cur_input=net_input[i*batchsize:(i+1)*batchsize]
                cur_input=cur_input.to(device)
                cur_output=(model(cur_input)).cpu()
                net_output=torch.cat((net_output,cur_output),axis=0)

    #print('net output:',net_output.shape)
    #exit(0)

    # net_output = torch.cat(output_list, dim=0)
    net_output=net_output.reshape(max_resolu,max_resolu,max_resolu)
    net_output=net_output.numpy()
    net_output=np.transpose(net_output,(2,1,0))

    ## set the SDF of the points too far away to be very high
    
    #TODO: load the gt mesh
    #tri_mesh = trimesh.load(f'common_3d_shapes/normalized_{shape_name}.obj')
    
    #mc_mask = gen_mc_mask(tri_mesh, net_input.numpy(), clamp_dist=0.1)
    #mc_mask = mc_mask.reshape(max_resolu,max_resolu,max_resolu)
    #mc_mask = np.transpose(mc_mask,(2,1,0))

    one_axis_points=net_output.shape[0]-1
    
    meshexport_list = []

    verts, faces, normals, values = measure.marching_cubes(volume=net_output,level=0,spacing=(length/one_axis_points,length/one_axis_points,length/one_axis_points))
    
    vertices = verts + np.array([-1.0, -1.0, -1.0])
    meshexport = trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=False)
    filename_save = "/home/lyang/code/lgy/patch_grid_siga/new_proj/_results/nglod_geo/" + f"{folder}.obj"
    meshexport.export(filename_save)
    print(f"{folder} saved")
    meshexport_list.append(meshexport)

    return meshexport_list

def re_mc(step_list,device,filename, model, max_resolu, extra_levels:list = None):
    
    levels=[0]
    if not extra_levels is None:
        assert isinstance(extra_levels, list)
        levels = levels + extra_levels

    
    low=-0.4
    high=0

    x_step=step_list[0]
    y_step=step_list[1]
    z_step=step_list[2]
    
    '''
    low=0
    high=0.4
    '''
    grid = np.linspace(low,high, max_resolu)

    #####################################
    """
    use this to replace the time-consuming for loop
    """
    xg, yg, zg = np.meshgrid(grid, grid, grid, indexing='ij')
    xg = xg.reshape(-1)
    yg = yg.reshape(-1)
    zg = zg.reshape(-1)

    net_input = np.stack((zg, yg, xg), axis=-1)
    net_input[:,0]=net_input[:,0]-x_step
    net_input[:,1]=net_input[:,1]-y_step
    net_input[:,2]=net_input[:,2]-z_step
    net_input=torch.FloatTensor(net_input)
    net_input=net_input.reshape((max_resolu*max_resolu*max_resolu,3))
    max_len=net_input.shape[0]
    batchsize=1000000

    with torch.no_grad():
        # output_list = []
        # i = 0
        # status = True
        # while status:
        #     start_id = i*batchsize
        #     end_id = (i+1)*batchsize
        #     if max_len < end_id:
        #         end_id = max_len
        #         status = False

        #     cur_input=net_input[start_id:end_id].cuda()
        #     cur_output=model(cur_input).cpu()
        #     output_list.append(cur_output)
        #     i = i+1

        cur_input=net_input[0:batchsize].to(device)
        net_output=model(cur_input)
        net_output=net_output.cpu()
        
        ## TODO: you can compute the max value of i by the following
        ## ceil(max_len / one_round_batchsize)
        for i in range(1,1000000): 
            if (i+1)*batchsize>max_len:
                cur_input=net_input[i*batchsize:max_len]
                cur_input=cur_input.to(device)
                cur_output=(model(cur_input)).cpu()
                net_output=torch.cat((net_output,cur_output),axis=0)
                break
            else:
                cur_input=net_input[i*batchsize:(i+1)*batchsize]
                cur_input=cur_input.to(device)
                cur_output=(model(cur_input)).cpu()
                net_output=torch.cat((net_output,cur_output),axis=0)


    # net_output = torch.cat(output_list, dim=0)
    net_output=net_output.reshape(max_resolu,max_resolu,max_resolu)
    net_output=net_output.numpy()
    net_output=np.transpose(net_output,(2,1,0))

    one_axis_points=net_output.shape[0]-1
    
    meshexport_list = []
    for lvl in levels:
        print(lvl)
        verts, faces, normals, values = measure.marching_cubes_lewiner(volume=net_output,level=lvl,spacing=(2/one_axis_points,2/one_axis_points,2/one_axis_points))
        vertices = verts + np.array([-1.0, -1.0, -1.0])
        meshexport = trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=False)
        filename_save = os.path.splitext(filename)[0] + f"_{lvl}.obj"
        meshexport.export(filename_save)
        print("saved")
        meshexport_list.append(meshexport)

    return meshexport_list


def save_gradients_at_gt_vertices(filename, vertices, normals, grad_size=1.0):
    # normals = normals/torch.norm(normals).reshape(-1, 1)
    write_edge_obj_file(filename, vertices, vertices+normals*grad_size)


def parse_args():
    parser = argparse.ArgumentParser(description="get geometry of an implicit function represented as a neural network")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--resume',
                        help='path to a trained model',
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    ## get config
    args = parse_args()

    resume_pth = args.resume
    assert os.path.isfile(resume_pth)

    config_path = args.cfg
    exp_type=config_path.split(".")[0]
    if os.path.isfile(config_path):
        config = read_json(config_path)
        #print("==== config ====")
        #for k, v in config.items():
        #    print(k, v)
        #print("==== config ====")
    else:
        print("no config file")
        exit()

    ## use cuda or not?
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print("device: ", device)

    mesh = trimesh.load("common_3d_shapes/normalized_fandisk.obj")
    if device != "cpu":
        vertices = torch.cuda.FloatTensor(mesh.vertices)
    else:
        vertices = torch.FloatTensor(mesh.vertices)
    
    sample_dir = config["data"]["file_path"]
    dataset = PolygonalMeshDataset(sample_dir,device=device)


    time_info=time.strftime("%m%d_%Hh%Mmin", time.localtime())
    exp_folder = os.path.join(config['trainer']['save_dir']+"_test", config['data']['shape_name'], time_info)
    res_dir = os.path.join(exp_folder, "geometry") ## save geometry

    ## build a folder to save everthing of this experiment
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    shutil.copyfile(config_path, os.path.join(res_dir, 'config.json')) ## copy the config file

    # ## load test data for computing chamfer distance
    # test_data=load_test_data(config['data']['shape_name'],config['trainer']['test_dir'])
    # print("load test data:", test_data.shape)

    net=Net(**config['network'])
    net = net.to(device)
    net.train()

    net.load_state_dict(torch.load(resume_pth, map_location=device))

    time1 = time.time()
    filename = "mesh_res_a.obj"
    marching_cube(filename, net, 256)
    time2 = time.time()
    print(time2-time1)

    
    filename = "mesh_res_normal_gt.obj"
    surface_grads = dataset.gradients[dataset.on_surface.reshape(-1),:]
    save_gradients_at_gt_vertices(
        filename, dataset.samples[dataset.on_surface.reshape(-1),:], surface_grads, grad_size=0.1)
    filename = "mesh_res_normal_pred.obj"
    vertices = dataset.samples[dataset.on_surface.reshape(-1),:]
    vertices.requires_grad = True
    pred_sdfs = net(vertices)
    normals = net.grad_compute(vertices, pred_sdfs)
    grad_diff_norm = torch.norm(normals + surface_grads, dim=1)
    print(grad_diff_norm.shape)
    print(grad_diff_norm.mean())


    save_gradients_at_gt_vertices(filename, vertices, -normals, grad_size=0.1)


    # time1 = time.time()
    # filename = os.path.join(res_dir, "mesh_res_b.ply")
    # my_marching_cube(filename, net, 256)
    # time2 = time.time()
    # print(time2-time1)