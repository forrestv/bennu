from __future__ import division

import base64
import json

from twisted.internet import defer
from twisted.python import log, util
from twisted.web import client, error

def getPageAndHeaders(url, contextFactory=None, *args, **kwargs):
    factory = client._makeGetterFactory(
        url,
        client.HTTPClientFactory,
        contextFactory=contextFactory,
        *args, **kwargs)
    return factory.deferred.addCallback(lambda page: (page, factory.response_headers))

class Error(Exception):
    def __init__(self, code, message, data=None):
        if not isinstance(code, int):
            raise TypeError('code must be an int')
        if not isinstance(message, unicode):
            raise TypeError('message must be a unicode')
        self._code, self._message, self._data = code, message, data
    def __str__(self):
        return '%i %s' % (self._code, self._message) + (' %r' % (self._data, ) if self._data is not None else '')
    def _to_obj(self):
        return {
            'code': self._code,
            'message': self._message,
            'data': self._data,
        }

class Proxy(object):
    def __init__(self, url, timeout=5, headers={}):
        self.url = url
        self.timeout = timeout
        self.headers = headers
    
    @defer.inlineCallbacks
    def callRemote(self, method, *params, **kwargs):
        if not set(['headers', 'receive_headers']).issuperset(kwargs.iterkeys()):
            raise ValueError()
       
        headers = util.InsensitiveDict({
            'Content-Type': 'application/json',
        })
        headers.update(self.headers)
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        
        try:
            data, response_headers = yield getPageAndHeaders(
                url=self.url,
                method='POST',
                agent=headers.get('User-Agent', 'Twisted JSON-RPC client'),
                headers=headers,
                postdata=json.dumps({
                    'jsonrpc': '2.0',
                    'method': method,
                    'params': params,
                    'id': 0,
                }),
                timeout=self.timeout,
            )
        except error.Error, e:
            try:
                resp = json.loads(e.response)
            except:
                raise e
            if 'error' in resp and resp['error'] is not None:
                raise Error(**resp['error'])
            raise e
        else:
            resp = json.loads(data)
            if 'error' in resp and resp['error'] is not None:
                raise Error(**resp['error'])
            if kwargs.get('receive_headers', False):
                defer.returnValue((resp['result'], util.InsensitiveDict(response_headers)))
            else:
                defer.returnValue(resp['result'])
    
    def __getattr__(self, attr):
        prefix = 'rpc_'
        if attr.startswith(prefix):
            return lambda *args, **kwargs: self.callRemote(attr[len(prefix):], *args, **kwargs)
        raise AttributeError('%r object has no attribute %r' % (self.__class__.__name__, attr))
