import caffe
import numpy as np


solver = caffe.SGDSolver("prototxt/solver.prototxt")
solver.net.copy_from("model/init.caffemodel")

caffe.set_device(0)
caffe.set_mode_gpu()

top_list = open("top_name.txt", 'r').readlines()
top_list = reversed(top_list)

solver.step(1)
iter = solver.iter
for each in top_list:
    each = each.strip("\n")
    print("------------------------------------- %s -------------------------" % each)
    print(solver.net.blobs[each].diff)



