I make some changes to your orginal code.

1. I add two new layers, the one is ImageSegDataLayer, which is used for the segmentation image and label.
The other one is InterpLayer, which is used to do bilinear interpolation. Because of this change, I have 
modified the caffe proto, common_layers.hpp and data_layers.hpp

2. For implementing the InterpLayer, I add four new files, interp.hpp, interp.cpp, interp.cu, common.cuh, 
which are used to help do bilinear interpolation

3. For the ImageSegDataLayer, I change the function ReadImageToCVMat in io.hpp and corresponding impleme-
ntation in io.cpp. At the same time, I add one function TransformImgAndSeg which is used to transform image 
and label simultaneously, so the data_transformer.hpp and data_transformer.cpp are both changed

4. In your bn_layer.cpp and sync_bn_layer.cpp, the shape of parameter blob is (1 x channel), but the shape 
in my initial model is (1 x channel x 1 x 1), so I make a small change in the LayerSetUp function of bn_layer.cpp 
and sync_bn_layer.cpp

All modifications on above are based on the caffe code of PSPNet, I change your code in order that 
I can use your SyncBNLayer to train one network which is based on ResNet101, I want the data in 
different GPUs can be used in the batch normalization computation.

In this directory, I put the necessray prototxts, image-label and init.caffemodel here, which you 
can use to reproduce the result. in prototxt directory, there are three pspnet101*.prototxt, they are used 
for caffe engine, cudnn engine and SyncBNLayer respectively. In the blob_check.py, I run step(1) for the 
solver, then I check the diff for all top blobs, if the engine is cudnn, you will find many nan in the printed 
messange, but if you use caffe engine, the result is normal 

