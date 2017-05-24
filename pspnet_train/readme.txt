I make some changes to the orginal yjxiong's code.

1. I add two new layers, the one is ImageSegDataLayer, which is used for the segmentation image and label.
The other one is InterpLayer, which is used to do bilinear interpolation. Because of this change, I have 
modified the caffe proto, common_layers.hpp and data_layers.hpp

2. For implementing the InterpLayer, I add four new files, interp.hpp, interp.cpp, interp.cu, common.cuh, 
which are used to help do bilinear interpolation

3. For the ImageSegDataLayer, I change the function ReadImageToCVMat in io.hpp and corresponding impleme-
ntation in io.cpp. At the same time, I add one function TransformImgAndSeg which is used to transform image 
and label simultaneously, so the data_transformer.hpp and data_transformer.cpp are both changed

4. In the bn_layer.cpp and sync_bn_layer.cpp, the shape of parameter blob is (1 x channel), but the shape 
in my initial model is (1 x channel x 1 x 1), so I make a small change in the LayerSetUp function of bn_layer.cpp 
and sync_bn_layer.cpp

All modifications on above are based on the caffe code of PSPNet, I change the code in order that 
I can use the SyncBNLayer to train one network which is based on ResNet101, I want the data in 
different GPUs can be used in the batch normalization computation.

If you want to use the code, in the root dir, you can do as below
    1. make build && cd build
    2. cmake -B. -H.. -DUSE_MPI=ON
    3. make && make install

In the code, I use Opencv3.0, so in cmake/Dependency.cmake, I set the opencv dir by hand, and I use cudnn 5 in the code

The inital model and evaluation model can be found here: http://pan.baidu.com/s/1eRYJHiq, the init.caffemodel is extracted from deeplab v2
