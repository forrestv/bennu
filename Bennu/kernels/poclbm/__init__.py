# Copyright (C) 2011 by jedi95 <jedi95@gmail.com> and 
#                       CFSworks <CFSworks@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pyopencl as cl
import numpy as np
import os
import math

import hashlib
from hashlib import md5
from struct import pack, unpack
from twisted.internet import defer, reactor, threads
from Midstate import calculateMidstate
#from KernelInterface import *
from BFIPatcher import *

def sleep(t):
    d = defer.Deferred()
    reactor.callLater(t, d.callback, None)
    return d

def _swap(s, l):
    return ''.join(s[x:x+l][::-1] for x in xrange(0, len(s), l))


class KernelData(object):
    """This class is a container for all the data required for a single kernel 
    execution.
    """
    
    def __init__(self, nonceRange, core, vectors, aggression):
        # Prepare some raw data, converting it into the form that the OpenCL
        # function expects.
        nonceRange.unit = nonceRange
        nonceRange.size = 2**32
        nonceRange.base = 0
        #print _swap(nonceRange.midstate, 4).encode('hex')
        nonceRange.midstate = calculateMidstate(_swap(nonceRange.data[:64], 4))
        #print nonceRange.midstate.encode('hex')
        data = np.array(
               unpack('IIII', _swap(nonceRange.unit.data[64:], 4)), dtype=np.uint32)
        
        # Vectors do twice the work per execution, so calculate accordingly...
        rateDivisor = 2 if vectors else 1
        
        # get the number of iterations from the aggression and size
        self.iterations = int((nonceRange.size / (1 << aggression)))
        self.iterations = max(1, self.iterations)
        
        #set the size to pass to the kernel based on iterations and vectors
        self.size = (nonceRange.size / rateDivisor) / self.iterations
        
        #compute bases for each iteration
        self.base = [None] * self.iterations
        for i in range(self.iterations):
            self.base[i] = pack('I',
                (nonceRange.base/rateDivisor) + (i * self.size))
        
        #set up state and precalculated static data
        self.state = np.array(
            unpack('IIIIIIII', nonceRange.unit.midstate), dtype=np.uint32)
        self.state2 = np.array(unpack('IIIIIIII',
            calculateMidstate(_swap(nonceRange.unit.data[64:80], 4) +
                '\x00\x00\x00\x80' + '\x00'*40 + '\x80\x02\x00\x00',
                nonceRange.unit.midstate, 3)), dtype=np.uint32)
        self.state2 = np.array(
            list(self.state2)[3:] + list(self.state2)[:3], dtype=np.uint32)
        self.nr = nonceRange
        
        self.f = np.zeros(8, np.uint32)
        self.calculateF(data)
        self.target = nonceRange.target
    
    def calculateF(self, data):
        rotr = lambda x,y: x>>y | x<<(32-y)
        self.f[0] = np.uint32(data[0] + (rotr(data[1], 7) ^ rotr(data[1], 18) ^
            (data[1] >> 3)))
        self.f[1] = np.uint32(data[1] + (rotr(data[2], 7) ^ rotr(data[2], 18) ^
            (data[2] >> 3)) + 0x01100000)
        self.f[2] = np.uint32(data[2] + (rotr(self.f[0], 17) ^
            rotr(self.f[0], 19) ^ (self.f[0] >> 10)))
        self.f[3] = np.uint32(0x11002000 + (rotr(self.f[1], 17) ^
            rotr(self.f[1], 19) ^ (self.f[1] >> 10)))
        self.f[4] = np.uint32(0x00000280 + (rotr(self.f[0], 7) ^
            rotr(self.f[0], 18) ^ (self.f[0] >> 3)))
        self.f[5] = np.uint32(self.f[0] + (rotr(self.f[1], 7) ^
            rotr(self.f[1], 18) ^ (self.f[1] >> 3)))
        self.f[6] = np.uint32(self.state[4] + (rotr(self.state2[1], 6) ^
            rotr(self.state2[1], 11) ^ rotr(self.state2[1], 25)) +
            (self.state2[3] ^ (self.state2[1] & (self.state2[2] ^
            self.state2[3]))) + 0xe9b5dba5)
        self.f[7] = np.uint32((rotr(self.state2[5], 2) ^
            rotr(self.state2[5], 13) ^ rotr(self.state2[5], 22)) +
            ((self.state2[5] & self.state2[6]) | (self.state2[7] &
            (self.state2[5] | self.state2[6]))))
        
        
class Kernel(object):
    """A Phoenix Miner-compatible kernel that uses the poclbm OpenCL kernel."""
    '''
    PLATFORM = KernelOption(
        'PLATFORM', int, default=None,
        help='The ID of the OpenCL platform to use')
    DEVICE = KernelOption(
        'DEVICE', int, default=None,
        help='The ID of the OpenCL device to use')
    VECTORS = KernelOption(
        'VECTORS', bool, default=False, advanced=True,
        help='Enable vector support in the kernel?')
    FASTLOOP = KernelOption(
        'FASTLOOP', bool, default=True, advanced=True,
        help='Run iterative mining thread?')
    AGGRESSION = KernelOption(
        'AGGRESSION', int, default=4, advanced=True,
        help='Exponential factor indicating how much work to run '
        'per OpenCL execution')
    WORKSIZE = KernelOption(
        'WORKSIZE', int, default=None, advanced=True,
        help='The worksize to use when executing CL kernels.')
    BFI_INT = KernelOption(
        'BFI_INT', bool, default=False, advanced=True,
        help='Use the BFI_INT instruction for AMD/ATI GPUs.')
    '''
    PLATFORM = None
    DEVICE = None
    VECTORS = False
    FASTLOOP = True
    AGGRESSION = 4
    WORKSIZE = None
    BFI_INT = False
    
    OUTPUT_SIZE = 0x100
    
    # This gets updated automatically by git.
    REVISION, = '$Id: 4c6684adb74c883590991651dfd2d7ff30fcfe4f $'.split(' ')[1:-1]
    
    def __init__(self, PLATFORM=None, DEVICE=None, VECTORS=None, FASTLOOP=False, AGGRESSION=4, WORKSIZE=None, BFI_INT=False):
        self.PLATFORM, self.DEVICE, self.VECTORS, self.FASTLOOP, self.AGGRESSION, self.WORKSIZE, self.BFI_INT = PLATFORM, DEVICE, VECTORS, FASTLOOP, AGGRESSION, WORKSIZE, BFI_INT
        if self.PLATFORM is not None: self.PLATFORM = int(self.PLATFORM)
        if self.DEVICE is not None: self.DEVICE = int(self.DEVICE)
        if self.AGGRESSION is not None: self.AGGRESSION = int(self.AGGRESSION)
        if self.WORKSIZE is not None: self.WORKSIZE = int(self.WORKSIZE)
        print self.DEVICE, repr(self.WORKSIZE)
        # Initialize object attributes and retrieve command-line options...)
        self.device = None
        self.kernel = None
        class X(object):
            def __getattr__(self, attr):
                def f(*args, **kwargs):
                    print attr, args, kwargs
                return f
        self.interface = X()
        self.core = self.interface.addCore()
        self.defines = ''
        self.loopExponent = 0
        
        # Set the initial number of nonces to run per execution
        # 2^(16 + aggression)
        self.AGGRESSION += 16
        self.AGGRESSION = min(32, self.AGGRESSION)
        self.AGGRESSION = max(16, self.AGGRESSION)
        self.size = 1 << self.AGGRESSION
        
        # We need a QueueReader to efficiently provide our dedicated thread
        # with work.
        #self.qr = QueueReader(self.core, lambda nr: self.preprocess(nr), 
        #                        lambda x,y: self.size * 1 << self.loopExponent)
        
        # The platform selection must be valid to mine.
        platforms = cl.get_platforms()

        if self.PLATFORM >= len(platforms) or \
            (self.PLATFORM is None and len(platforms) > 1):
            self.interface.log(
                'Wrong platform or more than one OpenCL platform found, '
                'use PLATFORM=ID to select one of the following\n',
                False, True)
            
            for i,p in enumerate(platforms):
                self.interface.log('    [%d]\t%s' % (i, p.name), False, False)
            
            # Since the platform is invalid, we can't mine.
            self.interface.fatal()
            return
        elif self.PLATFORM is None:
            self.PLATFORM = 0
            
        devices = platforms[self.PLATFORM].get_devices()
        print len(devices), self.DEVICE, self.DEVICE >= len(devices), (self.DEVICE is None and len(devices) > 1)
        
        # The device selection must be valid to mine.
        if self.DEVICE >= len(devices) or \
            (self.DEVICE is None and len(devices) > 1):
            self.interface.log(
                'No device specified or device not found, '
                'use DEVICE=ID to specify one of the following\n',
                False, True)
            
            for i,d in enumerate(devices):
                self.interface.log('    [%d]\t%s' % (i, d.name), False, False)
        
            # Since the device selection is invalid, we can't mine.
            self.interface.fatal()
            return
        elif self.DEVICE is None:
            self.DEVICE = 0
        
        self.device = devices[self.DEVICE]
        
        # We need the appropriate kernel for this device...
        try:
            self.loadKernel(self.device)
        except Exception:
            self.interface.fatal("Failed to load OpenCL kernel!")
            return
        
        # Initialize a command queue to send commands to the device, and a
        # buffer to collect results in...
        self.commandQueue = cl.CommandQueue(self.context)
        self.output = np.zeros(self.OUTPUT_SIZE+1, np.uint32)
        self.output_buf = cl.Buffer(
            self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
            hostbuf=self.output)
    
        self.applyMeta()
        
    def applyMeta(self):
        """Apply any kernel-specific metadata."""
        self.interface.setMeta('kernel', 'poclbm r%s' % self.REVISION)
        self.interface.setMeta('device', self.device.name.replace('\x00',''))
        self.interface.setMeta('cores', self.device.max_compute_units)
    
    def loadKernel(self, device):
        """Load the kernel and initialize the device."""
        self.context = cl.Context([device], None, None)
        
        # These definitions are required for the kernel to function.
        self.defines += (' -DOUTPUT_SIZE=' + str(self.OUTPUT_SIZE))
        self.defines += (' -DOUTPUT_MASK=' + str(self.OUTPUT_SIZE - 1))
        
        # If the user wants to mine with vectors, enable the appropriate code
        # in the kernel source.
        if self.VECTORS:
            self.defines += ' -DVECTORS'
        
        # Some AMD devices support a special "bitalign" instruction that makes
        # bitwise rotation (required for SHA-256) much faster.
        if (device.extensions.find('cl_amd_media_ops') != -1):
            self.defines += ' -DBITALIGN'
            #enable the expierimental BFI_INT instruction optimization
            if self.BFI_INT:
                self.defines += ' -DBFI_INT'
        else:
            #since BFI_INT requires cl_amd_media_ops, disable it
            if self.BFI_INT:
                self.BFI_INT = False
        
        # Locate and read the OpenCL source code in the kernel's directory.
        kernelFileDir, pyfile = os.path.split(__file__)
        kernelFilePath = os.path.join(kernelFileDir, 'kernel.cl')
        kernelFile = open(kernelFilePath, 'r')
        kernel = kernelFile.read()
        kernelFile.close()
        
        # For fast startup, we cache the compiled OpenCL code. The name of the
        # cache is determined as the hash of a few important,
        # compilation-specific pieces of information.
        m = md5()
        m.update(device.platform.name)
        m.update(device.platform.version)
        m.update(device.name)
        m.update(self.defines)
        m.update(kernel)
        cacheName = '%s.elf' % m.hexdigest()
        
        fileName = os.path.join(kernelFileDir, cacheName)
        
        # Finally, the actual work of loading the kernel...
        try:
            binary = open(fileName, 'rb')
        except IOError: 
            binary = None
        
        try:
            if binary is None:
                self.kernel = cl.Program(
                    self.context, kernel).build(self.defines)
                 
                #apply BFI_INT if enabled
                if self.BFI_INT:
                    #patch the binary output from the compiler
                    patcher = BFIPatcher(self.interface)
                    binaryData = patcher.patch(self.kernel.binaries[0])
                    
                    self.interface.debug("Applied BFI_INT patch")
                    
                    #reload the kernel with the patched binary
                    self.kernel = cl.Program(
                        self.context, [device],
                        [binaryData]).build(self.defines)
                
                #write the kernel binaries to file
                binaryW = open(fileName, 'wb')
                binaryW.write(self.kernel.binaries[0])
                binaryW.close()
            else:
                binaryData = binary.read()
                self.kernel = cl.Program(
                    self.context, [device], [binaryData]).build(self.defines)
                    
        except cl.LogicError:
            self.interface.fatal("Failed to compile OpenCL kernel!")
            return
        except PatchError:
            self.interface.fatal('Failed to apply BFI_INT patch to kernel! '
                'Is BFI_INT supported on this hardware?')
            return
        finally:
            if binary: binary.close()
       
        cl.unload_compiler()
        
        # If the user didn't specify their own worksize, use the maxium
        # supported by the device.
        maxSize = self.kernel.search.get_work_group_info(
                  cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
        
        if self.WORKSIZE is None:
            self.WORKSIZE = maxSize
        else:
            if self.WORKSIZE > maxSize:
                self.interface.log('Warning: Worksize exceeds the maximum of '
                                    + str(maxSize) + ', using default.')
            if self.WORKSIZE < 1:
                self.interface.log('Warning: Invalid worksize, using default.')
            
            self.WORKSIZE = min(self.WORKSIZE, maxSize)
            self.WORKSIZE = max(self.WORKSIZE, 1)
            #if the worksize is not a power of 2, round down to the nearest one
            if (self.WORKSIZE & (self.WORKSIZE - 1)) != 0:   
                self.WORKSIZE = 1 << int(math.floor(math.log(X)/math.log(2)))
            
        self.interface.setWorkFactor(self.WORKSIZE)
    
    def updateIterations(self):
        # Set up the number of internal iterations to run if FASTLOOP enabled
        rate = self.core.getRate()
         
        if not (rate <= 0):
            #calculate the number of iterations to run
            EXP = max(0, (math.log(rate)/math.log(2)) - (self.AGGRESSION - 8))
            #prevent switching between loop exponent sizes constantly
            if EXP > self.loopExponent + 0.54:
                EXP = round(EXP)
            elif EXP < self.loopExponent - 0.65:
                EXP = round(EXP)
            else:
                EXP = self.loopExponent
                
            self.loopExponent = int(max(0, EXP))
        
    def preprocess(self, nr):
        if self.FASTLOOP:
            self.updateIterations()
        
        kd = KernelData(nr, self.core, self.VECTORS, self.AGGRESSION)
        class Object(tuple):
            pass
        for i in xrange(kd.iterations):
            o = Object((kd, i))
            o.size = 2**32//kd.iterations
            yield o
    
    def postprocess(self, output, nr, got):
        # Scans over a single buffer produced as a result of running the
        # OpenCL kernel on the device. This is done outside of the mining thread
        # for efficiency reasons.
        
        # Iterate over only the first OUTPUT_SIZE items. Exclude the last item
        # which is a duplicate of the most recently-found nonce.
        for i in xrange(self.OUTPUT_SIZE):
            if output[i]:   
                data = nr.data[:76] + struct.pack(">I", output[i])
                hash_ = int(hashlib.sha256(hashlib.sha256(data).digest()).digest()[::-1].encode('hex'), 16)
                if hash_ <= nr.target:
                    got(data)
                else:
                    print 1, hash_//2**(256-32-32)
                    print 2, nr.target//2**(256-32-32)
    
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
                
                data, i = work

                self.kernel.search(
                    self.commandQueue, (data.size, ), (self.WORKSIZE, ),
                    data.state[0], data.state[1], data.state[2], data.state[3],
                    data.state[4], data.state[5], data.state[6], data.state[7],
                    data.state2[1], data.state2[2], data.state2[3],
                    data.state2[5], data.state2[6], data.state2[7],
                    data.base[i],
                    data.f[0], data.f[1], data.f[2], data.f[3],
                    data.f[4], data.f[5], data.f[6], data.f[7],
                    self.output_buf)
                cl.enqueue_read_buffer(
                    self.commandQueue, self.output_buf, self.output)
                #self.commandQueue.finish()
                yield threads.deferToThread(self.commandQueue.finish)

                # The OpenCL code will flag the last item in the output buffer
                # when it finds a valid nonce. If that's the case, send it to
                # the main thread for postprocessing and clean the buffer
                # for the next pass.
                if self.output[self.OUTPUT_SIZE]:
                    self.postprocess(self.output.copy(), data.nr, solution_putter)

                    self.output.fill(0)
                    cl.enqueue_write_buffer(
                        self.commandQueue, self.output_buf, self.output)
                    #for data in res:
                    #    solution_putter(data)

        work()
        return lambda: flag.__setitem__(0, False)
 
