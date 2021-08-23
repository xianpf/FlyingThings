# python run.py flying_objects.py
from genericpath import exists
import os, sys, importlib, shutil, cv2, json, glob
import random, h5py, argparse, subprocess
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from io import BytesIO

from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import all_images, read_image, check_lowpass, v2e_quit
from v2ecore.renderer import EventRenderer, ExposureMode


def convert_events(item_dir, preview=True, fps=60):
    rgb_dir = f"{item_dir}/rgb"
    h5py_dir = f"{item_dir}/h5py"
    flow_dir = f"{item_dir}/flow"
    events_dir = f"{item_dir}/events"
    evt_flow_dir = f"{item_dir}/evt_flow"
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(evt_flow_dir, exist_ok=True)

    rgb_H, rgb_W, _ = plt.imread(glob.glob(f'{rgb_dir}/*')[0]).shape
    # output_folder = 'generate_events/fromdata/output20'
    
    emulator = EventEmulator(
        pos_thres=0.15, 
        neg_thres=0.15,
        sigma_thres=0.03, 
        cutoff_hz=300,
        leak_rate_hz=0.01, 
        shot_noise_rate_hz=0.001,
        seed=0,
        output_folder=events_dir, 
        dvs_h5=None, 
        dvs_aedat2='blender.aedat',
        dvs_text='v2e-dvs-events.txt', 
        show_dvs_model_state=None,
        output_width=rgb_H, 
        output_height=rgb_W)

    eventRenderer = EventRenderer(
        output_path=events_dir,
        dvs_vid='dvs-video.avi', 
        preview=preview, 
        full_scale_count=2,
        exposure_mode=ExposureMode.DURATION,
        exposure_value=0.005,
        area_dimension=None,
        avi_frame_rate=30)
    eventRenderer.height = rgb_H
    eventRenderer.width = rgb_W


    events_list = []
    events_flow_list = []
    # import pdb; pdb.set_trace()
    nFrames = len(glob.glob(f'{h5py_dir}/*.hdf5'))
    # fps = 60
    # fps = 300
    interpTimes = np.linspace(0, nFrames/fps, nFrames, False).tolist()
    # array to batch events for rendering to DVS frames
    events_toshow = np.zeros((0, 4), dtype=np.float32)

    print(f'*** Stage 3/3: emulating DVS events from {nFrames} frames')
    with tqdm(total=nFrames, desc='dvs', unit='fr') as pbar:
        for i in range(nFrames):
            fr = read_image(f'{rgb_dir}/colors_{i:04d}.png')
            newEvents = emulator.generate_events(fr, interpTimes[i])
            pbar.update(1)
            if newEvents is not None and newEvents.shape[0] > 0:
                events_list.append(newEvents)
                with h5py.File(f'{h5py_dir}/{i-1}.hdf5', 'r') as h5pydata:
                    # print("hdf5 contains the following keys: " + str(h5pydata.keys()))
                    forward_flow = np.array(h5pydata['forward_flow'])
                    backward_flow = np.array(h5pydata['backward_flow'])
                    # import pdb; pdb.set_trace()
                    np.save(f'{flow_dir}/forward_flow_{i-1:04d}.npy', forward_flow)
                    np.save(f'{flow_dir}/backward_flow_{i-1:04d}.npy', backward_flow)
                evt_flow = forward_flow[newEvents[:,2].astype(np.int32), newEvents[:,1].astype(np.int32)]
                events_flow_list.append(evt_flow)

                # event_mask = np.zeros_like(evt_flow)

                events_toshow = np.append(events_toshow, newEvents, axis=0)
                events_toshow = np.array(events_toshow)

                # import pdb; pdb.set_trace()
                histrange = np.asarray([(0, v) for v in (eventRenderer.height, 
                                eventRenderer.height)], dtype=np.int64)
                eventRenderer.accumulate_event_frame(newEvents, histrange)
                newImg = (eventRenderer.currentFrame+eventRenderer.full_scale_count
                    )/float(eventRenderer.full_scale_count*2)
                eventRenderer.currentFrame = None
                cv2.imwrite(f"{events_dir}/evt_view_{i:04d}.png" , cv2.cvtColor(
                                (newImg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
                if i % 2 == 0:
                    eventRenderer.render_events_to_frames(events_toshow, height=rgb_H, width=rgb_W)

                    events_toshow = np.zeros((0, 4), dtype=np.float32)
        # process leftover events
        if len(events_toshow) > 0:
            eventRenderer.render_events_to_frames(
                events_toshow, height=rgb_H, width=rgb_W)
        events_np = np.concatenate(events_list, axis=0)
        events_flow = np.concatenate(events_flow_list, axis=0)
        print(np.unique(events_np[:, 0]))
        np.save(f'{events_dir}/events.npy', events_np)
        np.save(f'{evt_flow_dir}/events_flow.npy', events_flow)
    # p = subprocess.Popen(['ffmpeg', '-i', f'{events_dir}/dvs-video.avi', 
    #         '-c:v', 'copy', '-c:a', 'copy', '-y', f'{events_dir}/dvs-video.mp4'])
    # p.wait()
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{item_dir}/events/evt_view_*.png'"
                f" -c:v libx264 -pix_fmt yuv420p {item_dir}/events.mp4")

def post_events_after_blender():
    with open(f'.xpf_item_dir_path.txt', 'rt') as f:
        item_dir = f.read()
    fps = 60
    convert_events(item_dir, False, fps=fps)
    print(item_dir)
    # if __name__ != "__main__":
    val_evt_flow(item_dir, fps=fps)
    return item_dir

def warp_event_np(evt, flow, t_aim, size, byFrameOrSpeed=True, time_scale=1.0, fps=0.0):
    evt_img = np.zeros(size)
    if byFrameOrSpeed:
        x_warp = evt[:, 1] + flow[:, 0]
        y_warp = evt[:, 2] + flow[:, 1]
    else:
        # import pdb; pdb.set_trace()
        x_warp = evt[:, 1] + flow[:, 0] * (t_aim - evt[:, 0]) * time_scale
        y_warp = evt[:, 2] + flow[:, 1] * (t_aim - evt[:, 0]) * time_scale
    x_warp = x_warp.round().clip(0, size[1] - 1)
    y_warp = y_warp.round().clip(0, size[0] - 1)
    evt_warped = np.stack((evt[:,0], x_warp, y_warp, evt[:,3]), axis=1)
    return evt_warped

def events_to_image_np_v0(evt, size):
    evt_xy = evt[:, 1:3].astype(np.int32)
    evt_img = np.zeros(size)
    np.add.at(evt_img, tuple(evt_xy.transpose(1,0).tolist()), evt[:,3].tolist())
    return evt_img

def events_to_image_np(evt, size):
    evt_yx = tuple(evt[:, [2, 1]].astype(np.int32).transpose(1, 0).tolist())
    evt_img = np.zeros(size)
    np.add.at(evt_img, evt_yx, evt[:,3].tolist())
    return evt_img

def evt_img_enhance_contrast(evt_img):
    evt_img_contrast = np.zeros(evt_img.shape+(3,))
    # evt_img_contrast = np.zeros(evt_img.shape+(3,), np.uint8)
    evt_img_contrast[evt_img > 0] = [1, 0, 0]
    evt_img_contrast[evt_img < 0] = [0, 0, 1]
    return evt_img_contrast


def warp_rgb_img(img, flow):
    H, W = img.shape[:2]
    new_coords = flow.copy()
    new_coords[:,:,0] += np.arange(W)
    new_coords[:,:,1] += np.arange(H)[:,np.newaxis]
    # new_coords < 0
    x_over = np.where(new_coords[:,:,0] > (W - 1))
    new_coords[x_over] = np.stack(x_over).transpose(1,0)
    warp_img = cv2.remap(img, new_coords, None, cv2.INTER_LINEAR)

    return warp_img

def warp_flow(base_flow, delta_flow, size):
    H, W = size
    biased_coords = base_flow.copy().round()
    biased_coords[:,:,0] += np.arange(W)
    biased_coords[:,:,1] += np.arange(H)[:,np.newaxis]
    # new_coords < 0
    x_over = np.where((biased_coords[:,:,0] < 0) | (biased_coords[:,:,0] > (W - 1)))
    biased_coords[x_over] = np.stack(x_over[::-1]).transpose(1,0)
    y_over = np.where((biased_coords[:,:,1] < 0) | (biased_coords[:,:,1] > (H - 1)))
    biased_coords[y_over] = np.stack(y_over[::-1]).transpose(1,0)
    biased_coords_line = biased_coords.reshape(H*W, 2).astype(np.int32)

    warped_flow = base_flow + delta_flow[biased_coords_line[:, 1], biased_coords_line[:, 0]].reshape(H, W, 2)

    return warped_flow

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

def make_any_rgb_flow(item_dir, ts, te, t_aim, fps):
    rgb_dir = f"{item_dir}/rgb"
    flow_dir = f"{item_dir}/flow"
    warp_dir = f"{item_dir}/warp"
    im_s = int(ts * fps)
    im_e = int(te * fps)     # 结束图片, 结束点, 时间段不取到
    im_aim = int(t_aim * fps)
    warped_rgbs = []
    warped_flows = []
    combos = []
    for i in range(im_s, im_e):
        f_temp = BytesIO()
        rgb_init = plt.imread(f'{rgb_dir}/colors_{i:04d}.png')[:,:,:3]
        rgb_warp = rgb_init.copy()
        flow_init = plt.imread(f'{flow_dir}/forward_flow_{i:04d}.png')[:,:,:3]
        flow_warp = np.zeros_like(rgb_warp[:,:,:2])
        if i < im_aim:
            for j in range(i, im_aim):
                flow_j = np.load(f'{flow_dir}/forward_flow_{j:04d}.npy')
                rgb_warp = warp_rgb_img(rgb_warp, flow_j)
                flow_warp = warp_flow(flow_warp, flow_j, rgb_warp.shape[:2])
        elif i > im_aim:
            for j in reversed(range(im_aim+1, i+1)):
                flow_j = np.load(f'{flow_dir}/backward_flow_{j:04d}.npy')
                rgb_warp = warp_rgb_img(rgb_warp, flow_j)
                flow_warp = warp_flow(flow_warp, flow_j, rgb_warp.shape[:2])
        flow_warp_speed = flow_warp /(im_aim - i + 1e-15) * fps
        # flow_warp_img = flow_to_rgb(flow_warp).clip(0,1)
        # plt.imsave(f_temp, flow_warp_img, cmap='jet', format='jpg')
        # flow_warp_img_jet = (plt.imread(f_temp, format='jpg')[:,:,:3]/255).copy()
        # f_temp.close()
        # rgb_oneshot_flow_warp = warp_rgb_img(rgb_init, flow_warp)
        # warped_flows.append(flow_warp)
        flow_warp_img = flow_to_rgb(flow_warp_speed).clip(0,1)
        plt.imsave(f_temp, flow_warp_img, cmap='jet', format='jpg')
        flow_warp_img_jet = (plt.imread(f_temp, format='jpg')[:,:,:3]/255).copy()
        f_temp.close()
        rgb_oneshot_flow_warp = warp_rgb_img(rgb_init, flow_warp_speed)
        warped_flows.append(flow_warp_speed)
        combo_i = np.vstack((rgb_init, rgb_warp, flow_init, flow_warp_img_jet, rgb_oneshot_flow_warp))
        warped_rgbs.append(rgb_warp)
        combos.append(combo_i)
    combo_rgb_flow = np.hstack(combos)
    plt.imsave(f"{item_dir}/warp_combo_rgb_flow.png", combo_rgb_flow)
    
    return warped_rgbs, warped_flows
    
def val_evt_flow_v0(item_dir, fps):
    rgb_dir = f"{item_dir}/rgb"
    events_dir = f"{item_dir}/events"
    evt_flow_dir = f"{item_dir}/evt_flow"
    evt_img_dir = f"{item_dir}/evt_img"
    os.makedirs(evt_img_dir, exist_ok=True)
    events_p = f"{item_dir}/events/events.npy"
    evt_flow_p = f"{item_dir}/evt_flow/events_flow.npy"
    events = np.load(events_p)
    evt_flow = np.load(evt_flow_p)
    t_avg, t_min, t_max = events[:,0].mean(), events[:,0].min(), events[:,0].max()

    # import pdb; pdb.set_trace()
    im0 = plt.imread(f"{item_dir}/rgb/colors_{0:04d}.png")[:,:,:3]
    im1 = plt.imread(f"{item_dir}/rgb/colors_{0+1:04d}.png")[:,:,:3]
    # plt.imshow(im0)
    ts = 0
    te = 1 / fps
    # rgb_H, rgb_W, _ = plt.imread(glob.glob(f'{rgb_dir}/*')[0]).shape
    size = plt.imread(glob.glob(f'{rgb_dir}/*')[0]).shape[:2]
    event0 = events[(events[:,0] >= ts) & (events[:,0] < te)]
    evt_img_0 = evt_img_enhance_contrast(events_to_image_np(event0, size))
    plt.imsave(f'{evt_img_dir}/event_img_{0:04d}_before_warp.png', evt_img_0)
    evt_flow0 = evt_flow[(events[:,0] >= ts) & (events[:,0] < te)]
    warped_event0 = warp_event_np(event0, evt_flow0, ts, size)
    evt_img_warped_0 = evt_img_enhance_contrast(events_to_image_np(warped_event0, size))
    plt.imsave(f'{evt_img_dir}/event_img_{0:04d}_after_warp.png', evt_img_warped_0)
    img_combo = np.hstack((np.vstack((im0, im1)), np.vstack((evt_img_0, evt_img_warped_0))))
    plt.imsave(f'{evt_img_dir}/event_img_combo_{0:04d}.png', img_combo)
    var_evt, var_warp = np.var(evt_img_0), np.var(evt_img_warped_0)

    import pdb; pdb.set_trace()

def val_evt_flow(item_dir, fps):
    rgb_dir = f"{item_dir}/rgb"
    events_dir = f"{item_dir}/events"
    evt_flow_dir = f"{item_dir}/evt_flow"
    evt_img_dir = f"{item_dir}/evt_img"
    os.makedirs(evt_img_dir, exist_ok=True)
    events_p = f"{item_dir}/events/events.npy"
    evt_flow_p = f"{item_dir}/evt_flow/events_flow.npy"
    events = np.load(events_p)
    evt_flow = np.load(evt_flow_p)
    t_avg, t_min, t_max = events[:,0].mean(), events[:,0].min(), events[:,0].max()
    idx_start, idx_end = 0, 20
    im0 = plt.imread(f"{item_dir}/rgb/colors_{idx_start:04d}.png")[:,:,:3]
    im1 = plt.imread(f"{item_dir}/rgb/colors_{idx_end:04d}.png")[:,:,:3]
    ts = idx_start / fps
    te = idx_end / fps   # 1 / fps
    tm = (ts + te) / 2
    # size = (512, 512)
    size = plt.imread(glob.glob(f'{rgb_dir}/*')[0]).shape[:2]
    events_range = events[(events[:,0] >= ts) & (events[:,0] < te)]
    evt_flow_speed_range = evt_flow[(events[:,0] >= ts) & (events[:,0] < te)] * fps
    evt_img_before = evt_img_enhance_contrast(events_to_image_np(events_range, size))
    plt.imsave(f'{evt_img_dir}/event_img_{0:04d}_before_warp.png', evt_img_before)
    events_range_warped = warp_event_np(events_range, evt_flow_speed_range, tm, size, False, 1, fps)
    evt_img_warped = evt_img_enhance_contrast(events_to_image_np(events_range_warped, size))
    plt.imsave(f'{evt_img_dir}/event_img_{0:04d}_after_warp.png', evt_img_warped)
    img_combo = np.hstack((np.vstack((im0, im1)), np.vstack((evt_img_before, evt_img_warped))))
    plt.imsave(f'{evt_img_dir}/event_img_combo_{0:04d}.png', img_combo)
    rgbs, flows = make_any_rgb_flow(item_dir, ts, te, tm, fps)
    event_img_idx = events_range[:, 0].astype(np.int32)

    # import pdb; pdb.set_trace()
    flows_np = np.stack(flows)
    events_end2end_flow = flows_np[event_img_idx, events_range[:, 2].astype(np.int32), 
                events_range[:, 1].astype(np.int32)]
    events_end2end_warped = warp_event_np(events_range, events_end2end_flow, tm, size, False, 1, fps)
    evt_img_end2end_warped = evt_img_enhance_contrast(events_to_image_np(events_end2end_warped, size))
    plt.imsave(f'{evt_img_dir}/event_img_{0:04d}_end2end_warp.png', evt_img_end2end_warped)
    warp_combo = np.hstack((evt_img_before, evt_img_warped, evt_img_end2end_warped))
    plt.imsave(f'{evt_img_dir}/event_img_{0:04d}_before_after_end2end_warp.png', warp_combo)

    var_evt, var_warp, var_e2e = np.var(evt_img_before), np.var(evt_img_warped), np.var(evt_img_end2end_warped)
    var_compare = str(f"Varances are {var_evt:.4f}(before) / {var_warp:.4f}(after) / {var_e2e:.4f}(e2e)."
            f"\tdownscale {var_evt/var_warp:.4f}x(after) / {var_evt/var_e2e:.4f}x(e2e)")
    print(var_compare)
    with open(f'{evt_img_dir}/var_compare.txt', 'wt') as f:
        f.write(var_compare)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--item_dir', default='', help="The folder path to calculate the events")
    args = parser.parse_args()
    item_dir = args.item_dir[:-1] if args.item_dir.endswith('/') else args.item_dir
    if os.path.exists(args.item_dir):
        fps = 60
        convert_events(item_dir, fps=fps)
        val_evt_flow(item_dir, fps=fps)
    else:
        item_dir = post_events_after_blender()
        # val_evt_flow(item_dir, fps=fps)

if __name__ == "__main__":
    main()
