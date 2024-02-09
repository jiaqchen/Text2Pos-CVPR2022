import os
import plyfile
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

import multiprocessing as mp
import time

from tqdm import tqdm

import trimesh
from trimesh.sample import sample_surface

from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
# converting the 3RScan dataset to fit with Text2Pos

COLOR_MAP_20 = {
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    16: (219., 219., 141.),
    24: (255., 127., 14.),
    0: (0., 0., 0.), # unlabeled
}

MATERIAL_NAME = ['wooden', 'plastic', 'metal', 'glass', 'stone', 'leather', 'concrete', 'ceramic', 'brick', 'padded', 'cardboard', 'marbled', 'carpet', 'cork', 'velvet', 'unlabeled'] # 15 different materials

# TODO: 
# [ ] get the point cloud from 3RScan and separate them into each object for Object3d
#     [ ] each Cell has a list of Object3d
# [ ] get the poses from 3RScan and give them to a specific cell (do i just take the center of the cell for now? let's hope this doesn't matter for coarse localization)
#     [ ] each Pose has a coordinate in the cell, coordinate in the world, ID of the best associated cell, and a list of Descriptions of the pose in the context of the Best Cell
path_to_objects = '/home/julia/Documents/h_coarse_loc/data/3DSSG/objects.json'

def visualize_labels(u_index, labels, palette, out_name, loc='lower left', ncol=7):
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [palette[index * 3] / 255.0, palette[index * 3 + 1] / 255.0, palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    plt.figure()
    plt.axis('off')
    legend = plt.legend(frameon=False, handles=patches, loc=loc, ncol=ncol, bbox_to_anchor=(0, -0.3), prop={'size': 5}, handlelength=0.7)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_name, bbox_inches=bbox, dpi=300)
    plt.close()



def get_palette():
    palette = []
    for key, value in COLOR_MAP_20.items():
        palette.append(np.array(value))
    palette = np.concatenate(palette)
    return palette

def obtain_mapper(scene_name):
    remapper = np.ones(1500) * (255)
    mapping_file = json.load(open(path_to_objects))['scans']
    for elem in mapping_file:
        if elem['scan'] == scene_name:
            for obj in elem['objects']:
                if 'material' in obj['attributes'].keys() and len(obj['attributes']['material'])==1:
                    material = obj['attributes']['material'][0]
                    material_id = MATERIAL_NAME.index(material)
                    obj_id = int(obj['id'])
                    remapper[obj_id] = material_id
                    print(obj['label'], obj_id, obj['global_id'], obj['attributes']['material'])
    return remapper

def get_label(scene_id, object_id):
    # open /3RScan/scene_id/semseg.v2.json
    path_to_3rscan = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/'
    path_to_semseg = os.path.join(path_to_3rscan, scene_id, 'semseg.v2.json')
    with open(path_to_semseg) as f:
        data = json.load(f)
    for obj in data['segGroups']:
        if obj['objectId'] == object_id:
            return obj['label']
    return None

def export_mesh(name, v, f, c=None):
    if len(v.shape) > 2:
        v, f = v[0], f[0]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)

def export_mesh_as_pc(name, v, c=None):
    # export using trimesh
    mesh = trimesh.Trimesh(vertices=v, vertex_colors=c)
    mesh.export(name)


def find_nearest_point_and_labels(scene_id, p_xyz, objectId, coords_color, rgba):
    fn = scene_id
    fn = os.path.join(path_to_3rscan, fn,  'labels.instances.annotated.v2.ply')
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    coords_colors = np.ascontiguousarray(v[:, 3:6])/ 127.5 - 1
    objectIds = np.ascontiguousarray(v[:, 6]).astype(int)
    globalIds = np.ascontiguousarray(v[:, 7]).astype(int)
    nyu40s = np.ascontiguousarray(v[:, 8]).astype(int)
    eigen13s = np.ascontiguousarray(v[:, 9]).astype(int)
    rio27s = np.ascontiguousarray(v[:, 10]).astype(int)

    # for every point in xyz, find the closest coord in coords and use the index to get the objectId
    dist = np.linalg.norm(coords - p_xyz, axis=1)
    min_idx = np.argmin(dist)
    assert(objectIds[min_idx] == objectId)
    assert(np.all(coords_colors[min_idx] == coords_color))
    return p_xyz, objectId, coords_color, rgba, globalIds[min_idx], nyu40s[min_idx], eigen13s[min_idx], rio27s[min_idx]

def separate_pc_by_object(xyz, rgba, coords, coords_colors, objectId):
    assert(len(xyz[0]) == len(coords[0])) # both position
    assert(len(xyz) == len(rgba))
    assert(len(coords) == len(coords_colors) == len(objectId))
    xyz_with_objectId = []
    # for every point in xyz, find the closest coord in coords and use the index to get the objectId
    for i, p in enumerate(xyz):
        dist = np.linalg.norm(coords - p, axis=1)
        min_idx = np.argmin(dist)
        xyz_with_objectId.append((p, objectId[min_idx], coords_colors[min_idx], rgba[i]))
    return xyz_with_objectId

def get_cells_from_3rscan(scene_id):
    # Path to the labels.instances.annotated.v2.ply files
    # /home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/SCENE_ID/labels.instances.annotated.v2.ply
    path_to_3rscan = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/'

    path_to_scene = os.path.join(path_to_3rscan, scene_id)
    # path_to_ply = os.path.join(path_to_scene, 'labels.instances.annotated.v2.ply') # get the point cloud

    path_to_ply = os.path.join('/home/julia/Downloads', 'mesh.refined.v2.ply')
    path_to_semseg = os.path.join(path_to_scene, 'semseg.v2.json')                 # separate point cloud into objects
    path_to_mesh = os.path.join(path_to_scene, 'mesh.refined.v2.obj')
    path_to_texture = os.path.join(path_to_scene, 'mesh.refined_0.png')

    mesh = trimesh.load(path_to_mesh)
    mesh_colors = mesh.visual.to_color()
    points = mesh.vertices
    colors = mesh_colors.vertex_colors
    points = np.asarray(points)
    colors = np.asarray(colors)

    # trimesh sample_surface
    samples, face_index, colors = sample_surface(mesh, 100000, sample_color=True, seed=1)

    # save points and vertex colors as a point cloud
    export_mesh_as_pc(os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'mesh_sampled_pc_{}.ply'.format(scene_id)), samples, colors)

    xyz = np.array(samples)
    rgba = np.array(colors)
    assert(len(xyz) == len(rgba))
    rgb = rgba[:, :3]

    # For every scene, we want to have a list of objects that have rgb, xyz, and a label
    # And then somehow want to correspond this with the semantically segmented point cloud to get the individual objects

    # Getting the objectId and points per object
    fn = scene_id
    fn = os.path.join(path_to_3rscan, fn,  'labels.instances.annotated.v2.ply')
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    coords_colors = np.ascontiguousarray(v[:, 3:6])/ 127.5 - 1
    objectId = np.ascontiguousarray(v[:, 6]).astype(int)
    # semantic_label = get_label(scene_id, objectId)

    # print(semantic_label) TODO: add semantic label

    # Points separated per object
    points_separated = separate_pc_by_object(xyz, rgba, coords, coords_colors, objectId)

    points_colors = [p[2] for p in points_separated]
    points = [p[0] for p in points_separated]

    export_mesh_as_pc(os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'mesh_sampled_pc_separated_{}.ply'.format(scene_id)), points, points_colors)
    torch.save(points_separated, os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'points_separated_{}.pt'.format(scene_id)))

    #     objects = process_points_to_objs(points)

    #     # objects.append(Object3d(id=objectId,
    #     #                 instance_id=globalId,
    #     #                 xyz=xyz,
    #     #                 rgb=rgb,
    #     #                 label=semantic_label))
        
    #     cells.append(Cell(
    #         idx=1234567891, # if we have more cells per scene then need to change
    #         scene_name=scene_id,
    #         objects=objects,
    #         cell_size=None,
    #         bbox_w=None
    #     ))

    # if True: # do visualization
    #     # Go through cells and get the xyz, rgb into a point cloud
    #     for c in cells:
    #         objects = c.objects


    #     # original_mesh = o3d.io.read_triangle_mesh(path_to_ply)
    #     # vertices = np.asarray(original_mesh.vertices)
    #     # triangles = np.asarray(original_mesh.triangles)

    #     # export_mesh(os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'material_{}.ply'.format(scene_id)), vertices, triangles)

    #     # vertex_labels[vertex_labels==255.] = len(COLOR_MAP_20) - 1 
    #     # visualize_labels(list(np.unique(vertex_labels.astype(int))), MATERIAL_NAME, 
    #     #                                     get_palette(), os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'label_{}.jpg'.format(scene_id)), ncol=5)

    #     # vertex_labels = material_id.astype(int)

    return

def add_semantic_labels(scene_id):
    # open the .pt file, 'points_separated_{scene_id}.pt
    path_to_3rscan = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/'

    # open torch object
    points_separated = torch.load(os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'points_separated_{}.pt'.format(scene_id)))
    points_with_sem = []

    for i, point in enumerate(points_separated):
        p, objectId, coords_colors, rgba = point

        # p, objectId, coords_color, rgba, globalId, nyu40, eigen13, rio27 = find_nearest_point_and_labels(scene_id, p, objectId, coords_colors, rgba)
    
        semantic_label = get_label(scene_id, objectId)
        # if semantic_label is None:
        #     print(f'no semantic label for {objectId} in {scene_id}')

        # add semantic label to the point
        # points_with_sem.append((p, objectId, coords_color, rgba, globalId, nyu40, eigen13, rio27, semantic_label))
        points_with_sem.append((p, objectId, coords_colors, rgba, semantic_label))

    torch.save(points_with_sem, os.path.join('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'points_with_sem_{}.pt'.format(scene_id)))

def check_3rscan():
    path_to_3rscan = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/'
    scene_ids = os.listdir(path_to_3rscan)
    print(f'number of scenes in 3RScan: {len(scene_ids)}')
    path_to_scanscribe = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned'
    scene_ids_scanscribe = os.listdir(path_to_scanscribe)
    print(f'number of scenes in scanscribe: {len(scene_ids_scanscribe)}')
    for s in scene_ids_scanscribe:
        if s not in scene_ids:
            print(f'{s} not in 3RScan')
    assert(all(s in scene_ids for s in scene_ids_scanscribe))
    return True

if __name__ == '__main__':
    assert(check_3rscan())

    path_to_3rscan = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/'
    scene_ids = os.listdir(path_to_3rscan)

    # start counting time
    start = time.time()

    p = mp.Pool(processes=mp.cpu_count())
    # p.map(get_cells_from_3rscan, scene_ids)
    p.map(add_semantic_labels, scene_ids)
    p.close()
    p.join()

    # end counting time
    end = time.time()
    print(f'Finished in {end-start} seconds')


    # TODO: get all the other labels from the 3RScan dataset