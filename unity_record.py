from debug_gym import create_unity_workspace

import imageio
import numpy as np

class VideoRecorder(object):
    def __init__(self, height=480, width=480, fps=10):
        self.fps = fps
        self.frames = []

    def init(self, file_name, enabled=True):
        self.frames = []
        self.enabled = enabled

        self.writer = imageio.get_writer(file_name, fps=self.fps)

    def record(self, env, camera=None):
        if self.enabled:
            frame = env.render(mode='rgb_array')
            frame = np.moveaxis(frame, 0, -1)
            self.frames.append(frame)

            self.writer.append_data(frame)

    def save(self):
        if self.enabled:
            # imageio.mimsave(file_name, self.frames, fps=self.fps)
            # imageio.imwrite(file_name, self.frames)
            self.writer.close()

    def reset(self):
        self.frames = []

if __name__ == '__main__':
    executable = "/home/medcvr/yifei/medcvr-rl/dvrk_mlagents/builds/reach_mvd_3cam_v2/reach_mvd_3cam_v2.x86_64"
    unity_env, unity = create_unity_workspace(executable)

    outputfile = "/home/medcvr/yifei/thesis/output/test_recording.mp4"

    recorder = VideoRecorder()

    recorder.init(outputfile)
    unity.reset()
    recorder.record(unity)

    for i in range(100):
        action = [[0.1, -0.2]]
        unity.step(action)
        recorder.record(unity)

    recorder.save()

    unity.close()