import sys

from twisted.internet import reactor, task

from . import kernels, transports

def parse_args():
    urls = []
    miner_args = []
    for arg in sys.argv[1:]:
        if ':' in arg and '=' not in arg:
            urls.append(arg)
        elif ':' not in arg and '=' in arg:
            miner_args.append(arg)
        else:
            print "Confused about %r, exiting!" % (arg,)
            return
    miner_args2 = {}
    for a in miner_args:
        k, v = a.split('=', 1)
        if k in miner_args2:
            print 'Duplicate argument %r, exiting!' % (k,)
            return
        miner_args2[k] = v
    kernel_name = miner_args2.pop('KERNEL', 'python')
    return kernel_name, miner_args2, urls

def run():
    kernel_name, kernel_args, urls = parse_args()
    
    kernel = kernels.get(kernel_name)(**kernel_args)
    
    tps = [transports.get(url, kernel.preprocess) for url in urls]
    
    for tp in tps: tp.start()
    
    work_done = [0, 0]
    frames_done = [-1]
    def get_work():
        for tp in tps:
            work = tp.get_work()
            if work is not None:
                work_done[0] += work_done[1]
                frames_done[0] += 1
                work_done[1] = work.size
                return work
    def got_solution(data):
        for tp in tps: # XXX
            tp.send_solution(data)
    
    start_time = reactor.seconds()
    kernel.start(get_work, got_solution)
    
    def status():
        print "%ikH/s %ifps" % (.001*work_done[0]/(reactor.seconds() - start_time), frames_done[0]/(reactor.seconds() - start_time))
    task.LoopingCall(status).start(1)
    
    reactor.run()
