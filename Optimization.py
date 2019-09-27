from threading import Thread
from time import sleep


class Worker(Thread):

    def __init__(self, target, queue, *, name='Worker'):
        super().__init__()
        self.name = name
        self.queue = queue
        self._target = target
        self._stoped = False

    def run(self):
        #event.wait()
        while not self.queue.empty():
            realization = self.queue.get()
            if realization == 'Kill':
                self.queue.put(realization)
                self._stoped = True
                break
            self._target(realization)
             
    
    def join(self):
        while not self._stoped:
            sleep(0.000001)
            