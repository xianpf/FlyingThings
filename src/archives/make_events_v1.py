# python run.py flying_objects.py
import os, sys, importlib, shutil, cv2, json, glob
import random, h5py, argparse, subprocess
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import all_images, read_image, check_lowpass, v2e_quit
from v2ecore.renderer import EventRenderer, ExposureMode


def convert_events(item_dir, preview=True):
    rgb_dir = f"{item_dir}/rgb"
    h5py_dir = f"{item_dir}/h5py"
    events_dir = f"{item_dir}/events"
    evt_flow_dir = f"{item_dir}/evt_flow"
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(evt_flow_dir, exist_ok=True)

    HH, WW = 512, 512
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
        output_width=HH, 
        output_height=WW)

    eventRenderer = EventRenderer(
        output_path=events_dir,
        dvs_vid='dvs-video.avi', 
        preview=preview, 
        full_scale_count=2,
        exposure_mode=ExposureMode.DURATION,
        exposure_value=0.005,
        area_dimension=None,
        avi_frame_rate=30)
    eventRenderer.height = HH
    eventRenderer.width = WW


    events_list = []
    events_flow_list = []
    # import pdb; pdb.set_trace()
    nFrames = len(glob.glob(f'{h5py_dir}/*.hdf5'))
    fps = 60
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
                # import pdb; pdb.set_trace()
                evt_flow = forward_flow[newEvents[:,1].astype(np.int32), newEvents[:,2].astype(np.int32)]
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
                    eventRenderer.render_events_to_frames(events_toshow, height=HH, width=WW)

                    events_toshow = np.zeros((0, 4), dtype=np.float32)
        # process leftover events
        if len(events_toshow) > 0:
            eventRenderer.render_events_to_frames(
                events_toshow, height=HH, width=WW)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--item_dir', help="The folder path to calculate the events")
    args = parser.parse_args()
    item_dir = args.item_dir[:-1] if args.item_dir.endswith('/') else args.item_dir
    convert_events(item_dir)

def post_events_after_blender():
    with open(f'.xpf_item_dir_path.txt', 'rt') as f:
        item_dir = f.read()
    convert_events(item_dir, False)
    print(item_dir)

if __name__ == "__main__":
    main()