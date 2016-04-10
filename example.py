#!/usr/bin/env python

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np

inputsize = 100
hiddensize = 200
seqlength = 50
minibatch = 8
numlayers = 2
inputmode = 0
direction = 0
mode = 0
datatype = 0

handle = libcudnn.cudnnCreate()

rnndesc = libcudnn.cudnnCreateRNNDescriptor()
dropoutdesc = libcudnn.cudnnCreateDropoutDescriptor()
cudnnSetDropoutDescriptor(dropoutdesc, handle, 0, 0, 0, 0)
libcudnn.cudnnSetRNNDescriptor(rnndesc, hiddensize, seqlength, numlayers, 
					  dropoutdesc, inputmode, direction, mode, datatype)


xdescs = [libcudnn.cudnnCreateTensorDescriptor() for _ in xrange(seqlength)]
[libcudnn.cudnnSetTensorNdDescriptor(xdesc, 0, 3, [inputsize, minibatch, seqlength]) for xdesc in xdescs]

hxdesc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(hxdesc, 0, 3, [hiddensize, minibatch, numlayers])

cxdesc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(cxdesc, 0, 3, [hiddensize, minibatch, numlayers])

paramssize = libcudnn.cudnnGetRNNParamsSize(handle, rnndesc, xdescs)

wdesc = libcudnn.cudnnCreateFilterDescriptor()
libcudnn.cudnnSetFilterNdDescriptor(wdesc, 0, 0, 3, [paramssize, 1, 1])

ydescs = [libcudnn.cudnnCreateTensorDescriptor() for _ in xrange(seqlength)]
[libcudnn.cudnnSetTensorNdDescriptor(ydesc, 0, 3, [hiddensize, minibatch, seqlength]) for ydesc in ydescs]

hydesc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(hydesc, 0, 3, [hiddensize, minibatch, numlayers])

cydesc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(cydesc, 0, 3, [hiddensize, minibatch, numlayers])

workspacesize = libcudnn.cudnnGetRNNWorkspaceSize(handle, rnndesc, xdescs)

reservespacesize = libcudnn.cudnnGetRNNTrainingReserveSize(handle, rnndesc, xdescs)

libcudnn.cudnnRNNForwardTraining(handle, rnndesc, 
						xdescs, x, 
						hxdesc, hx, 
						cxdesc, cx, 
						wdesc, w, 
						ydesc, y, 
						hydesc, hy, 
						cydesc, cy, 
						workspace, workspacesize, 
						reservespace, reservespacesize)
