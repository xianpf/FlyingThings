
import argparse, glob, cv2, os
import numpy as np

parser = argparse.ArgumentParser("Script to visualize generated files")
parser.add_argument('-o', '--one_case', action='store_true', help='Determine if only one case or many cases.')
parser.add_argument('-d', '--obj_dir', help='Path to generated datasets file/s')

def stack_one_case(obj_dir, one_case):
    rgb_dir = obj_dir + "rgb"
    flow_dir = obj_dir + "flow"
    evt_dir = obj_dir + "events"
    stk_dir = obj_dir + "stack"
    if os.path.exists(f"{obj_dir}/stack.mp4"):
        return
    os.makedirs(stk_dir, exist_ok=True)
    for i in range(len(glob.glob(rgb_dir + '/colors_*.png'))-1):
        # import pdb; pdb.set_trace()
        rgb = cv2.imread(f"{rgb_dir}/colors_{i:04d}.png")
        fwd = cv2.imread(f"{flow_dir}/forward_flow_{i:04d}.png")
        bwd = cv2.imread(f"{flow_dir}/backward_flow_{i:04d}.png")
        evt = cv2.imread(f"{evt_dir}/evt_view_{i+1:04d}.png")
        combo = np.hstack((rgb, evt, fwd, bwd))
        cv2.imwrite(f"{stk_dir}/combo_{i:04d}.png", combo)
    print(f"ffmpeg -framerate 30 -pattern_type glob -i '{stk_dir}/combo_*.png'"
                f" -c:v libx264 -pix_fmt yuv420p {obj_dir}/stack.mp4")
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{stk_dir}/combo_*.png'"
                f" -c:v libx264 -pix_fmt yuv420p {obj_dir}/stack.mp4")

def main():
    args = parser.parse_args()
    one_case = True
    if args.one_case:
        args = parser.parse_args()
        stack_one_case(args.obj_dir, args.one_case)
    else:
        for obj_dir in glob.glob(args.obj_dir + '/*/'):
            print(obj_dir)
            if obj_dir.endswith('.txt'):
                continue
            try:
                stack_one_case(obj_dir, args.one_case)
            except:
                import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
    