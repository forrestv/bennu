def get(name):
    return __import__(name, globals(), locals(), [], 1).Kernel
