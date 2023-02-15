import time

class FPS:
    def __init__(self):
        self.start_time = None
        self.frames = 0
        self.fps = 0

    def start(self):
        self.start_time = time.time()

    def update(self):
        self.frames += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > 1:
            self.fps = self.frames / elapsed_time
            self.frames = 0
            self.start_time = current_time

    def get_fps(self):
        return self.fps
