import os
import shutil
import threading, queue
import logging
import time
import ffmpeg
import cv2
from scipy.io.wavfile import write
import torch
import torch.nn.functional as F


class VideoWriter(object):
    def __init__(self, cfg) -> None:
        super().__init__()

        if cfg.SYS.ASYNC_VIDEO_SAVING:
            self.q = queue.Queue()
            threading.Thread(target=self.worker).start()
    
    def worker(self):
        while True:
            print(self.q.qsize())
            func, args = self.q.get()
            func(*args)
            self.q.task_done()

    def save_video(self, cfg, tag, frames, step, epoch, global_step=None, long_img=None, audio=None, writer=None, base_path=None, extra_id=None):
        if 'tensorboard' in cfg.SYS.VIDEO_FORMAT:
            func = self.save_video_in_tensorboard
            args = (cfg, tag, frames, step, epoch, global_step, writer, extra_id)
            if cfg.SYS.ASYNC_VIDEO_SAVING:
                self.q.put((func, args))
            else:
                func(*args)
        if 'mp4' in cfg.SYS.VIDEO_FORMAT:
            func = self.save_video_in_mp4
            args = (cfg, tag, frames, step, epoch, global_step, audio, base_path, extra_id)
            if cfg.SYS.ASYNC_VIDEO_SAVING:
                self.q.put((func, args))
            else:
                func(*args)
        
        if 'img' in cfg.SYS.VIDEO_FORMAT:
            func = self.save_video_in_long_img
            args = (cfg, tag, long_img, step, epoch, global_step, base_path, extra_id)
            if cfg.SYS.ASYNC_VIDEO_SAVING:
                self.q.put((func, args))
            else:
                func(*args)

    def save_video_in_long_img(self, cfg, tag, long_img, step, epoch, global_step, base_path, extra_id=None):
        vid_tic = time.time()

        if tag != 'DEMO':
            return

        img_dir = os.path.join(base_path, 'imgs')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        img_path = '%s/epoch%d-%s-step%s.jpg' %(img_dir, epoch, tag, step) \
            if extra_id is None else '%s/epoch%d-%s-step%s-%d.jpg' %(img_dir, epoch, tag, step, extra_id)
        if os.path.exists(img_path):
            os.remove(img_path)
        cv2.imwrite(img_path, long_img)

        vid_toc = (time.time() - vid_tic)
        logging.info('[%s] epoch: %d/%d  step: %d  Saved %s in %.3f seconds.' % (
            tag, epoch, cfg.TRAIN.NUM_EPOCHS, step, 'long image', vid_toc))
    
    def save_video_in_tensorboard(self, cfg, tag, frames, step, epoch, global_step, writer, extra_id=None):
        vid_tic = time.time()

        if tag == 'TRAIN':  # training
            clip_tag = 'train/video'
            tb_step = global_step
        elif tag == 'VAL' or tag == 'TEST':
            clip_tag = '%s/video/%d' % (tag.lower(), step)
            tb_step = epoch
        elif tag == 'DEMO':
            return
        else:
            raise Exception('Unknown tag:' % tag)

        if extra_id is not None:
            clip_tag += f'/{extra_id}'

        with torch.no_grad():
            frames = torch.Tensor(frames).permute(0,3,1,2) / 255
            frames = torch.flip(frames, (1,))
            frames = F.interpolate(frames, scale_factor=0.4, mode='area')
            frames = torch.unsqueeze(frames, 0)
            writer.add_video(clip_tag, frames, tb_step, cfg.DATASET.FPS)

        vid_toc = (time.time() - vid_tic)
        logging.info('[%s] epoch: %d/%d  step: %d  Saved %s videos in %.3f seconds.' % (
            tag, epoch, cfg.TRAIN.NUM_EPOCHS, step, 'tensorboard', vid_toc))

    def save_video_in_mp4(self, cfg, tag, frames, step, epoch, global_step, audio, base_path, extra_id=None):
        vid_tic = time.time()

        vid_dir = os.path.join(base_path, 'videos')
        tmp_dir = os.path.join(vid_dir, 'tmp', '%f' % time.time())
        print("tmp_dir:", tmp_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.save_frames_in_jpg(frames, tmp_dir)

        global vid_path
        vid_path = '%s/epoch%d-%s-step%s.mp4' %(vid_dir, epoch, tag, step) \
            if extra_id is None else '%s/epoch%d-%s-step%s-%d.mp4' %(vid_dir, epoch, tag, step, extra_id)
        if os.path.exists(vid_path):
            os.remove(vid_path)
        
        import cv2

        video_name = os.path.join(tmp_dir,'output_video.mp4')
        fps = cfg.DATASET.FPS 
        print(fps)
        images = [img for img in os.listdir(tmp_dir) if img.endswith(".jpg")]
        # images.sort()  # Ensure images are in order

    # Determine the width and height from the first image
        frame = cv2.imread(os.path.join(tmp_dir, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print("writing images video")
        for image in images:
            video.write(cv2.imread(os.path.join(tmp_dir, image)))

        cv2.destroyAllWindows()
        video.release()
        exit()
# Usage
        # Frames per second

        # global input_video
        input_video = (
            ffmpeg
            .input('%s/*.jpg' % tmp_dir, pattern_type='glob', framerate=cfg.DATASET.FPS)
        )
        # print(input_video)
        # jpg_files = glob.glob(os.path.join(tmp_dir, '*.jpg'))
        # input_files = [os.path.join(tmp_dir, img) for img in os.listdir(tmp_dir) if img.endswith('.jpg')]
        # input_video = ffmpeg.concat(*[ffmpeg.input(file, framerate=cfg.DATASET.FPS) for file in input_files])
        # print(input_video)
        # import glob
        # jpg_files = glob.glob(os.path.join(tmp_dir, '*.jpg'))
        # input_video = (ffmpeg.input(file, framerate=cfg.DATASET.FPS) for file in jpg_files)
        
    # Concatenate input sources
        # input_video = ffmpeg.concat(*input_video, v=1, a=0)

        if audio is not None:
            wav_path = '%s/epoch%d-%s-step%s.wav' %(vid_dir, epoch, tag, step) \
            if extra_id is None else '%s/epoch%d-%s-step%s-%d.wav' %(vid_dir, epoch, tag, step, extra_id)
            write(wav_path, cfg.DATASET.AUDIO_SR, audio)
            global input_audio 
            input_audio = ffmpeg.input(wav_path)
            try:
                ffmpeg.concat(input_video, input_audio, v=1, a=1).output(vid_path).run(quiet=True)
            except ffmpeg.Error as e:
                print('An error occurred:', e.stderr)

        try:
            shutil.rmtree(tmp_dir)
        except:
            pass

        vid_toc = (time.time() - vid_tic)
        logging.info('[%s] epoch: %d/%d  step: %d  Saved %s videos in %.3f seconds.' % (
            tag, epoch, cfg.TRAIN.NUM_EPOCHS, step, 'mp4', vid_toc))

    def save_frames_in_jpg(self, frames, output_dir):
        print("saving images")
        for idx, frame in enumerate(frames):
            img_path = os.path.join(output_dir, '%06d.jpg' % idx)
            cv2.imwrite(img_path, frame)
    