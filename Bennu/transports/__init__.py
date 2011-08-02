def get(url, *args, **kwargs):
    scheme = url.split(':')[0].lower()
    mod = __import__(scheme, globals(), locals(), [], 1)
    return mod.Transport(url, *args, **kwargs)
