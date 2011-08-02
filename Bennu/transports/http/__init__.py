import base64
import urlparse

from twisted.internet import defer, reactor
from twisted.python import log

from . import jsonrpc

def sleep(t):
    d = defer.Deferred()
    reactor.callLater(t, d.callback, None)
    return d

def _swap(s, l):
    return ''.join(s[x:x+l][::-1] for x in xrange(0, len(s), l))

class Work(object):
    def __init__(self, j):
        if 'identifier' in j:
            self.identifier = 0, j['identifier']
        else:
            self.identifier = j['data'][:72]
        self.data = _swap(j['data'].decode('hex'), 4)[:80]
        self.target = int(j['target'].decode('hex')[::-1].encode('hex'), 16)
        self.midstate = _swap(j['midstate'].decode('hex'), 4)

def url_to_url_and_auth(url):
    if '@' not in url:
        return url, None
    scheme, rest = url.split('://', 2)
    userinfo, rest2 = rest.split('@', 2)
    return scheme + '://' + rest2, userinfo

class Transport(object):
    def __init__(self, url, preprocessor=lambda w: [w]):
        self.url, auth = url_to_url_and_auth(url)
        self.preprocessor = preprocessor
        
        self.headers = {'User-Agent': 'Bennu', 'X-All-Targets': '1', 'X-Work-Identifier': 1}
        if auth is not None:
            self.headers['Authorization'] = 'Basic ' + base64.b64encode(auth)
        
        self.long_polling_url = None
        self.seen_identifiers = set()
        self.current_identifier = None
        self.work = set()
    
    def start(self):
        self.work_filler()
        self.long_poller()
    
    def handle_response(self, r):
        w = Work(r)
        if w.identifier != self.current_identifier:
            if w.identifier in self.seen_identifiers:
                return
            else:
                self.work.clear()
                self.seen_identifiers.add(w.identifier)
                self.current_identifier = w.identifier
        self.work.add(self.preprocessor(w))
    
    @defer.inlineCallbacks
    def work_filler(self):
        while True:
            if len(self.work) >= 3:
                yield sleep(1)
            try:
                x, headers = yield jsonrpc.Proxy(self.url, headers=self.headers).rpc_getwork(receive_headers=True)
            except:
                log.err()
                yield sleep(1)
            else:
                self.handle_response(x)
                for url in headers.get('X-Long-Polling', []):
                    self.long_polling_url = urlparse.urljoin(self.url, url)
    
    @defer.inlineCallbacks
    def long_poller(self):
        while True:
            while self.long_polling_url is None:
                yield sleep(.1)
            print "Long polling..."
            try:
                x = yield jsonrpc.Proxy(self.long_polling_url, timeout=60, headers=self.headers).rpc_getwork()
            except defer.TimeoutError:
                print "Long poll timed out"
                pass
            except:
                print "Long poll errored"
                log.err()
                yield sleep(1)
            else:
                print "Long poll response"
                self.handle_response(x)
    
    def get_work(self):
        while True:
            if not self.work:
                return None
            x = self.work.pop()
            for item in x:
                self.work.add(x)
                return item
    
    @defer.inlineCallbacks
    def send_solution(self, solution):
        res = yield jsonrpc.Proxy(self.url, headers=self.headers).rpc_getwork(_swap(solution + '\x00'*48, 4).encode('hex'))
        print "Solution good:", res
        # XXX retry
