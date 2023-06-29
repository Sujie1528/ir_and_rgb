import math
import warnings
import os
import shutil
import time
import glob
from collections import deque
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
from threading import Thread

import cv2
import numpy as np

from btk import guideir, utils

# Here is the default colormaps list
cnames = list(utils.COLORMAPS.keys())
COLORMAP = [utils.COLORMAPS[c] for c in cnames]

# Here is the class of our camera stream object that being injected into this computer
class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        self.stream.set(cv2.CAP_PROP_BRIGHTNESS, 1000)
        self.stream.set(cv2.CAP_PROP_FOCUS, 10)
        self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.stream.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200)
        self.stream.set(cv2.CAP_PROP_EXPOSURE, -5)
        
        
        (self.grabbed, self.frame) = self.stream.read()
        self.name = name
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, name=self.name)
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def close(self):
        self.stopped = True

# helper function in case you need to point at one specific point
def _mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ir_y, ir_x = param["ir_yx"]
        if x >= ir_x:
            cx, cy = x - ir_x, y - ir_y
        else:
            cx, cy = x, y
        param["circle"] = utils.Circle(cx, cy, 5, size=param["size"])
        param["position"] = (cx, cy)
    
    
def get_idx(path):
    return int(os.path.splitext(path)[0].split("-")[-1])
    
# A class we will use when the stream didn't work well or we forgot to record it    
class Replayer:
    def __init__(self, dirname):
        if os.path.exists(dirname):
            if not os.path.isdir(dirname):
                raise ValueError("{} is not a directory".format(dirname))
        else:
            raise ValueError("Can't find {}".format(dirname))
            
        self.images = sorted([p for p in glob.glob(os.path.join(dirname, "visible/*.jpg"))], key=lambda p: get_idx(p))
        self.temps = sorted(glob.glob(os.path.join(dirname, "thermal/*.npy")), key=lambda p: get_idx(p))
        self.dts = [os.path.basename(p)[:23] for p in self.temps]
        self.idx = 0
        
    def read(self):
        image = cv2.imread(self.images[self.idx])
        temp = np.load(self.temps[self.idx])
        dt = datetime.strptime(self.dts[self.idx], "%Y-%m-%d-%H-%M-%S-%f")
        
        self.idx += 1
        return image, temp, dt
    
    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        temp = np.load(self.temps[i])
        return image, temp
    
    def __len__(self):
        return len(self.images)
                
# The general streamer class    
class Streamer:
    def __init__(self, src=0, delay=4):
        self.wf230 = guideir.PyWF230()
        if self.wf230.open() == 0:
            print("IR connected successfully")
        else:
            raise FileNotFoundError("Can't find IR")
            
        self.video = WebcamVideoStream(src).start()
        
        self.images = deque(maxlen=delay)
        self.idx = 0
        
    def read(self):
        dt = datetime.now()
        temp = self.wf230.read()
        
        self.images.append(np.flip(np.flip(self.video.read(), axis=0), axis=1))
        image = self.images[0]
       
        self.idx += 1
        return image, temp, dt
    
    def close(self):
        self.wf230.close()
        self.video.close()
        

def main(**kwargs):
    # Update the parameters given by the if __name__ == __main__ block
    globals().update(kwargs)
    upper = kwargs["upper"]
    lower = kwargs["lower"]
    #---------------------------------------------------------------------------
    
    # helper functions for checking the maximum temperature list
    def all_greater(lst):
        for e in lst:
            if e < 35:
                return False
        return True
    
    def all_lower(lst):
        for e in lst:
            if e > 35:
                return False
        return True
    
    # Check if we want to replay or go on streaming
    if replay_path is not None:
        mode = "replay"
        worker = Replayer(replay_path)
    else:
        mode = "stream"
        worker = Streamer(0, 4)
    
    if hasattr(worker, "wf230"):
        ih, iw = worker.wf230.get_height(), worker.wf230.get_width()
        grade = worker.wf230.get_temp_grade() - 1
        emissivity = worker.wf230.get_emissivity() - 1
        distance = worker.wf230.get_distance()
    else:
        ih, iw = 192, 256
        emissivity = 94
        distance = 2
        grade = 1
    
    grade = 1    
    h, w = 480, 640 
    
    # parameters that could be changed by keyboard or automatically 
    saving = 0
    render = 2
    recording = 0
    save_point = 0
    dynamic = 1
    fuse = 0
    colormap_idx = 0
    fah = 0
    color = (255, 255, 0)
    color2 = (0, 255, 0)
    
    param = {"size": (h, w), "ir_yx": (0, w), "position": None, "circle": None, "points": [[], []]}
    fps = -1
    data_id = save_id = video_id = point_id = 1
    videowriter = None
    ts_sum = time_sum = 0

    # parameters for the display window
    bar_h, bar_w = int(h * 2 / 3), int(w / 25)
    bar_top, bar_left = int((h - bar_h) / 2), 0
    uni = np.linspace(1, 0, bar_h).reshape(-1, 1).repeat(bar_w, axis=1)
    
    outline = np.uint8([[[180, 180, 180]]]).repeat(bar_h + 2, axis=0).repeat(bar_w + 2, axis=1)
    outline[1:bar_h + 1, 1:bar_w + 1] = utils.pseudo_color(uni, COLORMAP[colormap_idx])
    
    fontsize = min(int(h * 0.06), 24)
    fontStyle = ImageFont.truetype("./assets/HYSongYunLangHeiW-1.ttf", fontsize, encoding="utf-8")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # the coordination for remapping references
    txn1, tyn1 = np.load("./assets/xy_rgb3.npy")
    txn2, tyn2 = np.load("./assets/xy_ir3.npy")
    # time_grd = cur_time + 2
    worker.wf230.set_temp_grade(grade + 1)
    
    # the initialized temperature list and the recording status list
    temp_l = [0 for i in range(0, 150)]
    recording_lst = [0, 0]

    # Keep reading one frame
    while True:
        image, temp, dt = worker.read()
        cur_time = time.time()
        timestamp = dt.timestamp()
        
        millisecond = int(dt.microsecond * 0.001)
        
        max_ = temp.max()
        min_ = temp.min()
        

        if worker.idx == 1: # if it is the first time being read
            lts_rdr = timestamp
            time_cm = cur_time
            time_ems = cur_time
            time_dst = cur_time
            time_grd = cur_time
            time_save = cur_time - 3
            
            upper = max_
            lower = min_
            
            last_idx = worker.idx
            
            if mode == "replay":
                save_dir = os.path.join(replay_path, "replay")
            else:
                save_dir = os.path.join(save_root, dt.strftime("%Y-%m-%d-%H-%M-%S"))
            visible_dir = os.path.join(save_dir, "visible")
            thermal_dir = os.path.join(save_dir, "thermal")
            
            if not os.path.exists(visible_dir):
                os.makedirs(visible_dir)
            if not os.path.exists(thermal_dir):
                os.makedirs(thermal_dir)
        else: 
            interval = timestamp - lts_rdr
            if abs(interval) >= 1: # wait for a second so that worker.idx - last_idx is not just 1
                fps = (worker.idx - last_idx) / interval # calculate the current fps
                lts_rdr = timestamp
                last_idx = worker.idx
                
        if saving and worker.idx % save_freq == 0: # save original rgb image and temp matrix
            name = "{}-{:0>3d}-{}".format(dt.strftime("%Y-%m-%d-%H-%M-%S"), millisecond, data_id)
            cv2.resize(image, (2592,1944))
            cv2.imwrite(os.path.join(visible_dir, "{}.jpg".format(name)), image)
            np.save(os.path.join(thermal_dir, "{}.npy".format(name)), temp)
            data_id += 1
            
        if dynamic:
            upper = 0.97 * upper + 0.03 * max_
            lower = 0.97 * lower + 0.03 * min_
        
        # normalize the metric so that it can be used for colorizing 
        norm = ((temp - lower) / (upper - lower)).clip(min=0.0, max=1.0)
        ir = utils.pseudo_color(norm, COLORMAP[colormap_idx])
        ir = cv2.remap(ir, txn2, tyn2, interpolation=cv2.INTER_LINEAR)
        
        image2 = cv2.resize(image, (640, 480))
        image2 = cv2.remap(image2, txn1, tyn1, interpolation=cv2.INTER_LINEAR)
        
        # Thanks to the remapping above, two images could be fused together to be seen
        if fuse:
            gray1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
            
            _y1 = utils.highpass(gray1, 3)
            _y2 = utils.lowpass(gray2, 10)
            
            _, cb, cr = utils.ycbcr(*np.split(ir, 3, axis=2))
            
            _y = (_y2 + 1.5 * _y1)[:, :, np.newaxis]
            image2 = np.concatenate(utils.bgr(_y, cb, cr), axis=2).clip(min=0, max=255).astype(np.uint8)
        
        if render:
            ir[bar_top:bar_top + bar_h + 2, bar_left:bar_left + bar_w + 2] = outline
        # This new ir have already got colorbar by default  
        ir = Image.fromarray(ir)
        draw = ImageDraw.Draw(ir)
        
        if cur_time - time_cm < 3:
            draw.text((0, h - fontsize * 2.4), cnames[colormap_idx].replace("_", " "), color, font=fontStyle)
        if cur_time - time_ems < 3:
            draw.text((w - 7 * fontsize, h - fontsize * 3.6), "Emissivity:{:4.2f}".format((emissivity + 1) * 0.01), color, font=fontStyle)
        if cur_time - time_dst < 3:
            draw.text((w - 7 * fontsize, h - fontsize * 4.8), "Distance:{}".format(distance), color, font=fontStyle)
        if cur_time - time_grd < 3:
            draw.text((8 * fontsize, h - fontsize * 1.2), "{} temp grade".format("high" if grade else "common"), color, font=fontStyle)
        
        # Show the text on the display windows
        if render:
            draw.text((0, 0), "℃", color, font=fontStyle)
            draw.text((bar_left + bar_w + 3, bar_top - 0.5 * fontsize), "{:.1f}".format(upper), color, font=fontStyle) # ≥ 
            draw.text((bar_left + bar_w + 3, bar_top + bar_h - 0.5 * fontsize), "{:.1f}".format(lower), color, font=fontStyle) # ≤ ℃
            
            if render == 2:
                draw.text((5 * fontsize, 0), "{}".format(dt.strftime("%Y-%m-%d  %H:%M:%S")), color, font=fontStyle)
            elif render == 3:
                draw.text((w - 3 * fontsize, 0), "{:5.1f}".format(fps), color, font=fontStyle)
                draw.text((5 * fontsize, 0), 
                          "{}.{:0>3d}  {}".format(dt.strftime("%Y-%m-%d  %H:%M:%S"), millisecond, worker.idx), color, font=fontStyle)
            if not dynamic:
                draw.text((0, h - fontsize * 1.2), "{:.1f}～{:.1f}".format(min_, max_), color, font=fontStyle)
                
            ir_rec = np.array(ir)
        else:
            ir_rec = np.array(ir)
        
        if cur_time - time_save < 3:
            draw.text((w - 4 * fontsize, h - fontsize * 2.4), "{} saved".format(save_id - 1), color2, font=fontStyle)
        if saving:
            draw.text((w - 4 * fontsize, h - fontsize * 1.2), "saving...", color2, font=fontStyle)
        if recording:
            draw.text((w - 3 * fontsize, fontsize * 2.4), "REC", color2, font=fontStyle)
                
        if param["position"] is not None:
            cx, cy = param["position"]
            tx, ty = txn2[cy, cx], tyn2[cy, cx]
            point_temp = utils.post_bilinear(temp, (ih, iw), (tx, ty))
            draw.text((w - 3 * fontsize, h // 2), "{:5.1f}".format(point_temp), color, font=fontStyle)
            
            with open(os.path.join(save_dir, "point_temps.txt"), "a" if point_id > 1 else "w") as f:
                if point_id == 1:
                    f.write("Time, Temperature(℃)\n")
                    
                if point_id >= 2:
                    cur_point_ts = timestamp

                    ts_left = int(last_point_ts)
                    ts_right = int(cur_point_ts)
                    if ts_right - ts_left >= 1:
                        pred_temp = ((ts_right - last_point_ts) * point_temp + \
                        (cur_point_ts - ts_right) * last_point_temp) / (cur_point_ts - last_point_ts)

                        f.write("{}, {:.2f}\n".format(time.strftime("%H:%M:%S", time.localtime(ts_right)), pred_temp))
                        
                last_point_ts = timestamp
                last_point_temp = point_temp
                    
            point_id += 1
                
        ir = np.array(ir)
        
        if param["circle"] is not None:
            cc = param["circle"]
            ir[cc.ys, cc.xs] = color
            
        if mode == "replay":
            disp_time = time.time()
            if worker.idx > 1:
                time_sum += disp_time - last_time
                ts_sum += timestamp - last_ts
                if time_sum < ts_sum:
                    time.sleep(ts_sum - time_sum)
            last_ts = timestamp
            last_time = disp_time

        # show on the screen the image and ir that were being colorized and remapped    
        display = np.concatenate((image2, ir), axis=1)
        cv2.imshow("image and ir", display)
        cv2.setMouseCallback("image and ir", _mouse, param)
        
        if recording:
            record = np.concatenate((image2, ir_rec), axis=1)
            videowriter.write(record) #add this frame to video

        print(fps)
        
        # Update the highest temperature list
        temp_l.append(max_)
        temp_l = temp_l[-150:]
        if all_greater(temp_l[-5:]): # As long as the lastest 5 highest temps are higher than 100
            saving = 1
            if fps != -1:
                recording = 1
                recording_lst.append(1)
                recording_lst = recording_lst[-2:]
                if recording_lst == [0, 1]: # if the status of recording changed from off to on, begin recording
                    h_rec, w_rec = display.shape[:2]
                    
                    videowriter = cv2.VideoWriter(os.path.join(save_dir, "0a_{}.mp4".format(video_id)), fourcc, fps, (w_rec, h_rec), True)
                    video_id += 1       
        elif all_lower(temp_l): # only if all highest temps in 150 frames are lower than 80
            saving = 0
            recording = 0
            recording_lst.append(0)
            recording_lst = recording_lst[-2:]
            if recording_lst == [1, 0]: # if the status of recording changed from on to off, end recording
                recording = 0
                videowriter.release()
        
        # check the keyboard action
        key = cv2.waitKeyEx(1)
        if key != -1:
            if key in [ord("q"), 27]:
                if mode == "stream":
                    worker.close()
                    
                if videowriter is not None:
                    videowriter.release()
                    
                cv2.destroyAllWindows()
                break
                
            elif key in [ord("e"), ord("E")]:
                offset = 1 if key == ord("e") else -1
                emissivity = max(min(emissivity + offset, 98), 0)
                worker.wf230.set_emissivity(emissivity + 1)
                time_ems = cur_time
                
            elif key == ord("f"):
                fuse = 1 - fuse
            
            elif key == ord("F"):
                fah = 1 - fah
            elif key in [ord("h"), ord("H")]:
                offset = -1 if key == ord("h") else -1
                distance = min(max(distance + offset, 0), 50)
                worker.wf230.set_distance(distance)
                time_dst = cur_time
                
            elif key in [ord("t"), ord("T")]:
                offset = 1 if key == ord("t") else -1
                render = (render + offset) % 4
                
            elif key in [ord("u"), ord("U")]:
                dynamic = 0
                offset = 1 if key == ord("u") else -1
                upper += offset
                
            elif key in [ord("l"), ord("L")]:
                dynamic = 0
                offset = 1 if key == ord("l") else -1
                lower += offset
                
            elif key == ord("s"):
                name = save_id
                cv2.imwrite(os.path.join(save_dir, "{}.jpg".format(name)), image)
                cv2.imwrite(os.path.join(save_dir, "{}_ir.jpg".format(name)), ir)
                np.save(os.path.join(save_dir, "{}.npy".format(name)), temp)
                time_save = cur_time
                save_id += 1
                
            # elif key == ord("m"):
            #     saving = 1 - saving
                
            elif key == ord("d"):
                dynamic = 1 - dynamic
                
            elif key in [ord("c"), ord("C")]:
                offset = 1 if key == ord("c") else -1
                colormap_idx = (colormap_idx + offset) % len(COLORMAP)
                
                time_cm = cur_time
                outline[1:bar_h + 1, 1:bar_w + 1] = utils.pseudo_color(uni, COLORMAP[colormap_idx])
            
            elif key == 3014656: # delete
                param["position"] = None
                param["circle"] = None
            elif mode == "replay":
                if key in [2555904, 2424832]:
                    skip = 80
                    if key == 2555904:
                        worker.idx = min(worker.idx + skip, len(worker) - 1)
                        last_ts = datetime.strptime(worker.dts[worker.idx - 1], "%Y-%m-%d-%H-%M-%S-%f").timestamp()
                    else:
                        worker.idx = max(worker.idx - skip, 1)
                        last_ts = datetime.strptime(worker.dts[worker.idx - 1], "%Y-%m-%d-%H-%M-%S-%f").timestamp()
                else:
                    print(key)
            else:
                    print(key)
                    
        if mode == "replay":
            if worker.idx == len(worker):
                cv2.destroyAllWindows()
                if videowriter is not None:
                    videowriter.release()
                break
                
    if len(os.listdir(visible_dir)) == 0:
        os.rmdir(visible_dir)
    if len(os.listdir(thermal_dir)) == 0:
        os.rmdir(thermal_dir)
    if len(os.listdir(save_dir)) == 0:
        os.rmdir(save_dir)
        

if __name__ == "__main__":
    main(
        save_root="./data",
        lower=20, 
        upper=30,
        save_freq=1, # the lower, the frequecy is actually higher (not quite sure, since it's not linear and depends on how quickly one loop goes, just my understanding)
        #replay_path=r"D:\ir_and_rgb\ir_and_rgb\data\2023-06-19-12-53-05"   #change this directory and uncomment this line if you want to replay
        replay_path=None
    )