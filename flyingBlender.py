# python run.py flying_objects.py
import os, sys
sys.path.append('blender/blender-2.92.0-linux64/custom-python-packages')
from src.utility.SetupUtility import SetupUtility
SetupUtility.setup_pip(['opencv-python', 'h5py', 'matplotlib'])

import importlib, shutil, cv2, json, glob, random
import bpy, h5py
from mathutils import Matrix, Vector, Euler
import numpy as np
from matplotlib import pyplot as plt
# sys.path.append('blender/blender-2.92.0-linux64/custom-python-packages')
# sys.path.append('/home/SENSETIME/xianpengfei/blender/blender-2.92.0-linux64/custom-python-packages')
# sys.path.append('/home/SENSETIME/xianpengfei/Works/BlenderProc-main')
# from src.utility.SetupUtility import SetupUtility
# SetupUtility.setup_pip(['opencv-python'])

from src.utility.loader.ObjectLoader import ObjectLoader
from src.utility.Utility import Utility
from src.utility.Config import Config
from src.main.InitializerModule import InitializerModule
from src.utility.loader.ShapeNetLoader import ShapeNetLoader
from src.utility.MaterialUtility import Material
from src.lighting.LightLoader import LightLoader
from src.camera.CameraLoader import CameraLoader
from src.renderer.FlowRenderer import FlowRenderer
from src.renderer.RgbRenderer import RgbRenderer
from src.writer.Hdf5Writer import Hdf5Writer


def sample_shape_and_bgimg(base_dir, shape_dir, bg_dir):
    saved_shape_p = f'{base_dir}/saved_shapes_p.txt'
    if os.path.exists(saved_shape_p):
        with open(saved_shape_p, 'rt') as f_shape:
            all_shapes_p = [s.strip() for s in f_shape.readlines()]
    else:
        # all_shapes_p = glob.glob(f"/home/xianr/data/datasets/ShapeNetCore.v2/*/*")
        all_shapes_p = glob.glob(f"{shape_dir}*/*")
        if not all_shapes_p:
            import pdb; pdb.set_trace()
        with open(saved_shape_p, 'wt') as f_shape:
            f_shape.write('\n'.join(all_shapes_p))
    shape_p = random.choice(all_shapes_p)
    synset_id, source_id = shape_p.split('/')[-2:]

    saved_bg_img_p = f'{base_dir}/saved_bg_img_p.txt'
    if os.path.exists(saved_bg_img_p):
        with open(saved_bg_img_p, 'rt') as f_shape:
            all_bg_img_p = [s.strip() for s in f_shape.readlines()]
    else:
        # all_bg_img_p = glob.glob(f"/home/xianr/data/datasets/ADE20K/images/ADE/training/*/*/*.jpg")
        all_bg_img_p = glob.glob(f"{bg_dir}images/ADE/training/*/*/*.jpg")
        if not all_bg_img_p:
            import pdb; pdb.set_trace()
        with open(saved_bg_img_p, 'wt') as f_shape:
            f_shape.write('\n'.join(all_bg_img_p))
    bg_img_p = random.choice(all_bg_img_p)

    uni_id = f"{synset_id}_{source_id}_{'_'.join(bg_img_p[:-4].split('/')[-3:])}"
    item_dir = f'{base_dir}/{uni_id}'

    return synset_id, source_id, bg_img_p, item_dir


def load_shapes(shape_dir, synset_id, source_id):
    # for shape_p in all_shapes_p:
    loaded_obj = ShapeNetLoader.load(
                data_path=shape_dir,
                used_synset_id=synset_id,
                used_source_id=source_id)[0]
    return loaded_obj

def load_bg_img(obj, bg_img_p, item_dir):
    materials = Material.convert_to_materials([obj.material_slots[0].material])

    bg_imgx1 = cv2.imread(bg_img_p)
    bgH, bgW, bgC = bg_imgx1.shape
    bg_imgx12 = bg_imgx1[None].repeat(12, axis=0).reshape(4,3,bgH,bgW,bgC
            ).transpose(0,2,1,3,4).reshape(4*bgH,3*bgW,bgC)
    bg_imgx12_p = f'{item_dir}/bg_imgx12.jpeg'
    cv2.imwrite(bg_imgx12_p, bg_imgx12)

    bpy.ops.image.open(filepath=bg_imgx12_p, directory=os.path.dirname(bg_imgx12_p))
    materials[0].set_principled_shader_value('Base Color', bpy.data.images.get(os.path.basename(bg_imgx12_p)))

def more_pose_frames(obj):
    bpy.context.scene.frame_end = 10
    for fi in range(10):
        obj.rotation_euler.x -= 0.8/10
        obj.keyframe_insert(data_path='rotation_euler', frame=fi)

def pose_cam_keyframes(obj, total_frames = 10, pose_key_frames=None, 
                        cam_key_frames=None, kf_num=4):
    bpy.context.scene.frame_end = total_frames - 1
    if pose_key_frames is None:
        pose_key_frames = [] # (frame,    loc_x, loc_y, loc_z,    rot_x, rot_y, rot_z)
        f_is = np.linspace(0, total_frames, kf_num).astype(np.int32).tolist()
        for i in range(kf_num):
            f_i = f_is[i]
            loc_x = random.uniform(-0.25, 0.25)
            loc_y = random.uniform(-0.1, 0.1)
            loc_z = random.uniform(-0.25, 0.25)
            rot_x = random.uniform(-0.8, 0.8)
            rot_y = random.uniform(-0.8, 0.8)
            rot_z = random.uniform(-0.8, 0.8)
            pose_key_frames.append((
                f_i,
                loc_x, loc_y, loc_z,
                rot_x, rot_y, rot_z))

    for i in range(len(pose_key_frames)-1):
        step0 = pose_key_frames[i]
        step1 = pose_key_frames[i+1]
        num = step1[0] - step0[0]
        # steps = np.linspace(step0[0], step1[0], num, False).tolist()
        fs = np.linspace(step0[0], step1[0], num, False).tolist()
        xs = np.linspace(step0[1], step1[1], num, False).tolist()
        ys = np.linspace(step0[2], step1[2], num, False).tolist()
        zs = np.linspace(step0[3], step1[3], num, False).tolist()
        rxs = np.linspace(step0[4], step1[4], num, False).tolist()
        rys = np.linspace(step0[5], step1[5], num, False).tolist()
        rzs = np.linspace(step0[6], step1[6], num, False).tolist()
        for f, x, y, z, rx, ry, rz in zip(fs, xs, ys, zs, rxs, rys, rzs):
            try:
                print(f)
                obj.location = Vector((x, y, z))
                obj.rotation_euler = Euler((rx, ry, rz), 'XYZ')
                obj.keyframe_insert(data_path='rotation_euler', frame=f)
            except:
                import pdb; pdb.set_trace()

    if cam_key_frames is None:
        cam_key_frames = [] # (frame,    loc_x, loc_y, loc_z,    rot_x, rot_y, rot_z)
        f_is = np.linspace(0, total_frames, kf_num).astype(np.int32).tolist()
        for i in range(kf_num):
            f_i = f_is[i]
            loc_x = random.uniform(0.1, 1.9)     # 1
            loc_y = random.uniform(-3.5, -2.5)       # -3
            loc_z = random.uniform(-0.5, 0.5)     # 0
            rot_x = random.uniform(1.48, 1.5)       # 1.5
            rot_y = random.uniform(1.48, 1.5)       # 1.5
            rot_z = random.uniform(0.25, 0.35)       # 0.3
            # loc_x = 0     # 1
            # loc_y = -40       # -3
            # loc_z = 0     # 0
            # rot_x = 1.5       # 1.5
            # rot_y = 1.5       # 1.5
            # rot_z = 0       # 0.3
            cam_key_frames.append((
                f_i,
                loc_x, loc_y, loc_z,
                rot_x, rot_y, rot_z))

    cam_ob = bpy.context.scene.camera
    for i in range(len(cam_key_frames)-1):
        step0 = cam_key_frames[i]
        step1 = cam_key_frames[i+1]
        num = step1[0] - step0[0]
        # steps = np.linspace(step0[0], step1[0], num, False).tolist()
        fs = np.linspace(step0[0], step1[0], num, False).tolist()
        xs = np.linspace(step0[1], step1[1], num, False).tolist()
        ys = np.linspace(step0[2], step1[2], num, False).tolist()
        zs = np.linspace(step0[3], step1[3], num, False).tolist()
        rxs = np.linspace(step0[4], step1[4], num, False).tolist()
        rys = np.linspace(step0[5], step1[5], num, False).tolist()
        rzs = np.linspace(step0[6], step1[6], num, False).tolist()
        for f, x, y, z, rx, ry, rz in zip(fs, xs, ys, zs, rxs, rys, rzs):
            print(f)
            cam_ob.location = Vector((x, y, z))
            cam_ob.rotation_euler = Euler((rx, ry, rz), 'XYZ')
            cam_ob.keyframe_insert(data_path='rotation_euler', frame=f)

def decode_h5py(item_dir):
    args_rgb_keys = ["colors", "normals", "diffuse"]
    args_flow_keys = ["forward_flow", "backward_flow"]
    args_other_non_rgb_keys = ["distance", "depth"]
    args_segmap_keys = ["segmap"]
    args_segcolormap_keys = ["segcolormap"]
    args_keys = None
    rgb_dir = f"{item_dir}/rgb"
    h5py_dir = f"{item_dir}/h5py"
    opt_flow_dir = f"{item_dir}/flow"
    output_dir = f'{item_dir}/blender_out'
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(h5py_dir, exist_ok=True)
    os.makedirs(opt_flow_dir, exist_ok=True)

    def flow_to_rgb(flow):
        """
        Visualizes optical flow in hsv space and converts it to rgb space.
        :param flow: (np.array (h, w, c)) optical flow
        :return: (np.array (h, w, c)) rgb data
        """
        im1 = flow[:, :, 0]
        im2 = flow[:, :, 1]

        h, w = flow.shape[:2]

        # Use Hue, Saturation, Value colour model
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[..., 1] = 1

        mag, ang = cv2.cartToPolar(im1, im2)
        hsv[..., 0] = ang * 180 / np.pi
        hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def vis_data(key, data, full_hdf5_data, file_label, save_name=''):
        # If key is valid and does not contain segmentation data, create figure and add title
        if not save_name and key in args_flow_keys + args_rgb_keys + args_other_non_rgb_keys:
            plt.figure()
            plt.title("{} in {}".format(key, file_label))

        if key in args_flow_keys:
            # Visualize optical flow
            if save_name:
                plt.imsave(save_name, flow_to_rgb(data).clip(0,1), cmap='jet')
            else:
                plt.imshow(flow_to_rgb(data), cmap='jet')
        elif key in args_segmap_keys:
            # Try to find labels for each channel in the segcolormap
            channel_labels = {}
            if args_segmap_keys.index(key) < len(args_segcolormap_keys):
                # Check if segcolormap_key for the current segmap key is configured and exists
                segcolormap_key = args_segcolormap_keys[args_segmap_keys.index(key)]
                if segcolormap_key in full_hdf5_data:
                    # Extract segcolormap data
                    segcolormap = json.loads(np.array(full_hdf5_data[segcolormap_key]).tostring())
                    if len(segcolormap) > 0:
                        # Go though all columns, we are looking for channel_* ones
                        for colormap_key, colormap_value in segcolormap[0].items():
                            if colormap_key.startswith("channel_") and colormap_value.isdigit():
                                channel_labels[int(colormap_value)] = colormap_key[len("channel_"):]

            # Make sure we have three dimensions
            if len(data.shape) == 2:
                data = data[:, :, None]
            # Go through all channels
            for i in range(data.shape[2]):
                # Try to determine label
                if i in channel_labels:
                    channel_label = channel_labels[i]
                else:
                    channel_label = i

                if save_name:
                    plt.imsave(save_name, data[:, :, i], cmap='jet')
                else:
                    # Visualize channel
                    plt.figure()
                    plt.title("{} / {} in {}".format(key, channel_label, file_label))
                    plt.imshow(data[:, :, i], cmap='jet')

        elif key in args_other_non_rgb_keys:
            # Make sure the data has only one channel, otherwise matplotlib will treat it as an rgb image
            if len(data.shape) == 3:
                if data.shape[2] != 1:
                    print("Warning: The data with key '" + key + "' has more than one channel which would not allow using a jet color map. Therefore only the first channel is visualized.")
                data = data[:, :, 0]
        
            if save_name:
                plt.imsave(save_name, data, cmap='jet')
            else:
                plt.imshow(data, cmap='jet')
        elif key in args_rgb_keys:
            if save_name:
                plt.imsave(save_name, data)
            else:
                plt.imshow(data)

    def vis_file(path, rgb_dir, opt_flow_dir):
        # import pdb; pdb.set_trace()
        img_idx = int(path[:-5].split('/')[-1])
        # Check if file exists
        if os.path.exists(path):
            if os.path.isfile(path):
                with h5py.File(path, 'r') as data:
                    print(path + " contains the following keys: " + str(data.keys()))

                    # Select only a subset of keys if args_keys is given
                    if args_keys is not None:
                        keys = [key for key in data.keys() if key in args_keys]
                    else:
                        keys = [key for key in data.keys()]

                    # Visualize every key
                    for key in keys:
                        value = np.array(data[key])
                        if key in ['colors']:
                            my_save_name = f"{rgb_dir}/{key}_{img_idx:04d}.png"
                        elif key in ['backward_flow', 'forward_flow', ]:
                            my_save_name = f"{opt_flow_dir}/{key}_{img_idx:04d}.png"
                        else:
                            my_save_name = "没什么用"
                        # Check if it is a stereo image
                        if len(value.shape) >= 3 and value.shape[0] == 2:
                            # Visualize both eyes separately
                            for i, img in enumerate(value):
                                vis_data(key, img, data, os.path.basename(path) + (" (left)" if \
                                    i == 0 else " (right)"), save_name=my_save_name)
                        else:
                            vis_data(key, value, data, os.path.basename(path), save_name=my_save_name)
                        # plt.savefig(f'{path[:-5]}_{key}.png')

            else:
                print("The path is not a file")
        else:
            print("The file does not exist")

    hdf5_paths = glob.glob(f'{output_dir}/*.hdf5')
    for path in hdf5_paths:
        vis_file(path, rgb_dir, opt_flow_dir)
        shutil.move(path, f"{h5py_dir}/{os.path.basename(path)}")
    for fn in glob.glob(f"{output_dir}/*.png")+glob.glob(f"{output_dir}/*.exr")+glob.glob(f"{output_dir}/*.npy"):
        os.remove(fn)
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{rgb_dir}/colors*.png'"
                f" -c:v libx264 -pix_fmt yuv420p {item_dir}/rgb.mp4")
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{opt_flow_dir}/forward_flow*.png'"
                f" -c:v libx264 -pix_fmt yuv420p {item_dir}/forward_flow.mp4")
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{opt_flow_dir}/backward_flow*.png'"
                f" -c:v libx264 -pix_fmt yuv420p {item_dir}/backward_flow.mp4")

def process_one(total_frames=10):
    # base_dir = '/data/FlyingThings'
    base_dir = '/home/xianr/data/FlyingThings'
    # base_dir = 'tmp'
    shape_dir = '/home/xianr/data/datasets/ShapeNetCore.v2/'
    bg_dir = '/home/xianr/data/datasets/ADE20K/'
    os.makedirs(base_dir, exist_ok=True)
    synset_id, source_id, bg_img_p, item_dir = sample_shape_and_bgimg(
                            base_dir, shape_dir, bg_dir)
    if os.path.exists(item_dir):
        return
    os.makedirs(item_dir, exist_ok=True)
    output_dir = f'{item_dir}/blender_out'
    Utility.temp_dir = Utility.resolve_path(output_dir)

    init_m = InitializerModule(Config({
            "output_dir": output_dir
        }))
    ligh_m = LightLoader(Config({
            "lights": [
                {
                    "type": "POINT",
                    "location": [5, -5, 5],
                    "energy": 1000
                }
            ],
            "output_dir": output_dir
        }))
    # cam_m = CameraLoader(Config(
        # {
        #     "cam_poses": [
        #         {
        #             "location": [1, -3, 0],
        #             #   "rotation": {
        #             #       "format": "look_at",
        #             #       "value": {
        #             #         "provider": "getter.POI"
        #             #       }
        #             #     }
        #             "rotation": {
        #                 "value":[1.5, 1.5, 0.3]
        #                 # "value":[1, 0, 0]
        #             }
        #         },
        #     ],
        #     "intrinsics": {
        #     "fov": 1
        #     },
        #     "output_dir": output_dir
        # }))
    flow_m = FlowRenderer(Config({
            "forward_flow_output_key": "forward_flow",
            "backward_flow_output_key": "backward_flow",
            "forward_flow": True,
            "backward_flow": True,
            "blender_image_coordinate_style": False,  # per default, the coordinate system origin is in the top left of the image
            "output_dir": output_dir
        }))
    rend_m = RgbRenderer(Config({
            "output_key": "colors",
            "samples": 350,
            "output_dir": output_dir
        }))
    writ_m = Hdf5Writer(Config({
            "module": "writer.Hdf5Writer",
            "output_dir": output_dir
        }))

    init_m.run()

    board_obj = ObjectLoader.load('board4.obj')[0]
    loaded_obj = load_shapes(shape_dir, synset_id, source_id)
    load_bg_img(board_obj.blender_obj, bg_img_p, item_dir)
    # more_pose_frames(loaded_obj.blender_obj)
    pose_cam_keyframes(loaded_obj.blender_obj, total_frames)

    ligh_m.run()
    # cam_m.run()
    flow_m.run()
    rend_m.run()
    writ_m.run()

    decode_h5py(item_dir)
    print(item_dir)

    # with open(f'.xpf_item_dir_path.txt', 'wt') as f:
    with open(f'{sys.argv[9]}/.xpf_item_dir_path.txt', 'wt') as f:
        f.write(item_dir)



if __name__ == "__main__":
    process_one(200)