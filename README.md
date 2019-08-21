# 个性化对象检测训练—— Pet Detector

学习 Google Tensorflow Object Detection API的例子

## 对象检测的训练和执行过程

训练过程经历了下面这个流程，通过 SGD 对 loss 进行优化。

    inputs (images tensor) -> preprocess -> predict -> loss -> outputs (loss tensor)

执行过程的流程如下：

    inputs (images tensor) -> preprocess -> predict -> postprocess ->
      outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)
      
## 目录结构

    /                      项目根目录
    |- README.md               
    |- data                训练所需的原始图片和素材
        |- pets
        |- starwar                   
    |- models              预训练好的模型文件下载   
        |- faster_rcnn_resnet101_coco_2018_01_28 
    |- object_detection    对象检测API，来自于 https://github.com/tensorflow/models/tree/master/research 
    |- training_data       本次训练所需的训练数据放置于此 
        |- pets
            |- faster_rcnn_resnet101_pets.config
            |- model.ckpt.data-00000-of-00001
            |- model.ckpt.index
            |- model.ckpt.meta
            |- pet_faces_train.record-*-of-*
            |- pet_label_map.pbtxt
            |- ...
        |- starwar
    |- venv                python venv 环境

## 安装Protoc, 编译 ProtoBuffer 模型

Protoc用于编译相关程序运行文件.

MAC 平台使用 brew 安装 protoc ，安装的是最新版。

    $ brew install protobuf
    $ protoc --version
    libprotoc 3.9.0

linux Ubuntu 直接用 apt install protobuf-compiler 安装的是 libprotoc 3.0.0， 所以我们需要从源代码编译最新的版本    
    
    $ sudo apt-get install autoconf automake libtool curl make g++ unzip
    
    $ wget https://github.com/protocolbuffers/protobuf/releases/download/v3.9.1/protobuf-all-3.9.1.tar.gz
    $ tar -xvf protobuf-all-3.9.1.tar.gz
    $ cd protobuf-all-3.9.1
    $ ./configure
    $ make
    $ make check
    $ sudo make install
    $ ln -s /usr/local/bin/protoc /usr/bin/protoc
    $ sudo ldconfig # refresh shared library cache.
    $ protoc --version
    libprotoc 3.9.1
    
编译Object Detection API的代码。成功后，能够在 protos 目录下看到编译出的 *_pb2.py 文件：

    $ protoc object_detection/protos/*.proto --python_out=.
    
测试安装，通过运行以下命令来测试您是否正确安装了Tensorflow Object Detection API：

    $ python object_detection/builders/model_builder_test.py

    WARNING: Logging before flag parsing goes to stderr.
    W0821 11:01:01.009028 140734862828992 lazy_loader.py:50] 
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    W0821 11:01:01.260434 140734862828992 deprecation_wrapper.py:119] From /Users/chenhao/workspaces/python/my_ai_study/venv/lib/python3.6/site-packages/slim-0.1-py3.6.egg/nets/inception_resnet_v2.py:373: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    W0821 11:01:01.320055 140734862828992 deprecation_wrapper.py:119] From /Users/chenhao/workspaces/python/my_ai_study/venv/lib/python3.6/site-packages/slim-0.1-py3.6.egg/nets/mobilenet/mobilenet.py:397: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.
    
    Running tests under Python 3.6.4: /Users/chenhao/workspaces/python/my_ai_study/venv/bin/python
    [ RUN      ] ModelBuilderTest.test_create_faster_rcnn_model_from_config_with_example_miner
    [       OK ] ModelBuilderTest.test_create_faster_rcnn_model_from_config_with_example_miner
    [ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
    [       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
    [ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
    [       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
    [ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
    [       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
    [ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
    [       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
    [ RUN      ] ModelBuilderTest.test_create_rfcn_model_from_config
    [       OK ] ModelBuilderTest.test_create_rfcn_model_from_config
    [ RUN      ] ModelBuilderTest.test_create_ssd_fpn_model_from_config
    [       OK ] ModelBuilderTest.test_create_ssd_fpn_model_from_config
    [ RUN      ] ModelBuilderTest.test_create_ssd_models_from_config
    [       OK ] ModelBuilderTest.test_create_ssd_models_from_config
    [ RUN      ] ModelBuilderTest.test_invalid_faster_rcnn_batchnorm_update
    [       OK ] ModelBuilderTest.test_invalid_faster_rcnn_batchnorm_update
    [ RUN      ] ModelBuilderTest.test_invalid_first_stage_nms_iou_threshold
    [       OK ] ModelBuilderTest.test_invalid_first_stage_nms_iou_threshold
    [ RUN      ] ModelBuilderTest.test_invalid_model_config_proto
    [       OK ] ModelBuilderTest.test_invalid_model_config_proto
    [ RUN      ] ModelBuilderTest.test_invalid_second_stage_batch_size
    [       OK ] ModelBuilderTest.test_invalid_second_stage_batch_size
    [ RUN      ] ModelBuilderTest.test_session
    [  SKIPPED ] ModelBuilderTest.test_session
    [ RUN      ] ModelBuilderTest.test_unknown_faster_rcnn_feature_extractor
    [       OK ] ModelBuilderTest.test_unknown_faster_rcnn_feature_extractor
    [ RUN      ] ModelBuilderTest.test_unknown_meta_architecture
    [       OK ] ModelBuilderTest.test_unknown_meta_architecture
    [ RUN      ] ModelBuilderTest.test_unknown_ssd_feature_extractor
    [       OK ] ModelBuilderTest.test_unknown_ssd_feature_extractor
    ----------------------------------------------------------------------
    Ran 16 tests in 0.147s
    
    OK (skipped=1)

## 基于宠物数据的训练

### 数据准备

需要从网上下载所需的数据。

    # From ai_pet_detector/
    $ cd data/pets
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    $ wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    $ tar -xvf images.tar.gz
    $ tar -xvf annotations.tar.gz

将数据转化成 Tensorflow Object Detection API 所需要的 TFRecord 模式。
 
    # From ai_pet_detector/
    $ python object_detection/dataset_tools/create_pet_tf_record.py \
        --label_map_path=object_detection/data/pet_label_map.pbtxt \
        --data_dir=`pwd`/data/pets \
        --output_dir=`pwd`/training_data/pets

复制 pet_label_map.pbtxt 到 training_data/pets 目录下

    # From ai_pet_detector/
    $ cp object_detection/data/pet_label_map.pbtxt training_data/pets/

### 从训练好的模型进行迁移学习

    # From ai_pet_detector/
    $ cd model
    $ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    $ tar -xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    
    $ cp faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.* ../training_data/pets
    $ cp object_detection/samples/configs/faster_rcnn_resnet101_pets.config training_data/pets/

修改 faster_rcnn_resnet101_pets.config 文件，将其中的 PATH_TO_BE_CONFIGURED 替换成您的数据所在目录。
在此例中，PATH_TO_BE_CONFIGURED 替换成 'training_data/pets'

    $ vi training_data/pets/faster_rcnn_resnet101_pets.config
    
### 执行训练

    $ python object_detection/model_main.py \
        --pipeline_config_path=training_data/pets/faster_rcnn_resnet101_pets.config \
        --model_dir=training_data/pets \
        --num_train_steps=50000 \
        --sample_1_of_n_eval_examples=1 \
        --alsologtostderr
              
## 基于千年隼和钛战机数据的训练

### 数据准备

    # From ai_pet_detector 的上级目录
    $ git clone https://github.com/bourdakos1/Custom-Object-Detection.git
    $ cd Custom-Object-Detection
    $ cp -r annotations ../ai_pet_detector/data/starwar
    $ cp -r images ../ai_pet_detector/data/starwar

将数据转化成 Tensorflow Object Detection API 所需要的 TFRecord 模式。
这里我们需要先修改出一个 create_starwar_tf_record.py。根据标注数据，最后修改的结果如 [create_starwar_tf_record.py](object_detection/dataset_tools/create_starwar_tf_record.py) 
修改出一个 faster_rcnn_resnet101_starwar.config。根据标注数据，最后修改的结果如 [faster_rcnn_resnet101_starwar.config](object_detection/samples/configs/faster_rcnn_resnet101_starwar.config) 

    # From ai_pet_detector/
    $ python object_detection/dataset_tools/create_starwar_tf_record.py \
        --label_map_path=object_detection/data/pet_label_map.pbtxt \
        --data_dir=`pwd`/data/starwar \
        --output_dir=`pwd`/training_data/starwar

    $ cp models/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.* training_data/starwar
    $ cp object_detection/samples/configs/faster_rcnn_resnet101_starwar.config training_data/starwar/

修改 faster_rcnn_resnet101_starwar.config 文件，将其中的 PATH_TO_BE_CONFIGURED 替换成您的数据所在目录。
在此例中，PATH_TO_BE_CONFIGURED 替换成 'training_data/starwar'

    $ vi training_data/pets/faster_rcnn_resnet101_pets.config
    
训练
    
    $ python object_detection/model_main.py \
        --pipeline_config_path=training_data/starwar/faster_rcnn_resnet101_starwar.config \
        --model_dir=training_data/starwar \
        --num_train_steps=50000 \
        --sample_1_of_n_eval_examples=1 \
        --alsologtostderr