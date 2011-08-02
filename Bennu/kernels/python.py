import hashlib
import struct

from twisted.internet import defer, reactor, threads

def sleep(t):
    d = defer.Deferred()
    reactor.callLater(t, d.callback, None)
    return d

def _do_work(work):
    # takes work, returns list of matching nonces
    good = []
    for i in xrange(work.nonce_start, work.nonce_end):
        data = work.data[:76] + struct.pack("<I", i)
        hash_ = int(hashlib.sha256(hashlib.sha256(data).digest()).digest()[::-1].encode('hex'), 16)
        if hash_ <= work.target:
            good.append(data)
    return good

class MyWork(object):
    def __init__(self, data, nonce_start, nonce_end, target):
        self.data, self.nonce_start, self.nonce_end, self.target = data, nonce_start, nonce_end, target
        self.size = nonce_end - nonce_start

class Kernel(object):
    def __init__(self): # could have options
        pass
    
    def preprocess(self, work):
        for i in xrange(0, 2**32, 2**12):
            yield MyWork(work.data, i, i + 2**12, work.target)
    
    def start(self, work_getter, solution_putter):
        # returns a callable to stop it
        flag = [True]
        @defer.inlineCallbacks
        def work():
            while flag[0]:
                work = work_getter()
                if work is None:
                    print "Worker starved!"
                    yield sleep(1)
                    continue
                res = yield threads.deferToThread(_do_work, work)
                for data in res:
                    solution_putter(data)
        work()
        return lambda: flag.__setitem__(0, False)
