from timeit import default_timer as timer

class Timer:
    """ Timer class that allows multiple timinigs """
    
    def __init__(self):
        self.t0 = None
        self.measurements = []
    
    def time(self, name=''):
        t1 = timer()
        self.measurements.append((name, t1 - self.t0)) # type: ignore
        self.t0 = timer()
        
    def __enter__(self):
        self.t0 = timer()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        for name, dt in self.measurements:
            print(f'{dt:.2f}s : {name}')