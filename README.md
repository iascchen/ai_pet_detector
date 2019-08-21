# 个性化对象检测训练—— Pet Detector

## 目录结构

    /                   项目根目录
    README.md               
    data                训练所需的原始图片和素材                   
    models              预训练好的模型文件下载    
    object_detection    对象检测API    
    training_data       本次训练所需的    
    venv


## 安装Protoc

Protoc用于编译相关程序运行文件.

### MAC 平台

使用brew安装protoc，安装的是最新版。

    $ brew install protobuf
    $ protoc --version
    libprotoc 3.9.0
    
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

千年隼和钛战机数据

    git clone https://github.com/bourdakos1/Custom-Object-Detection.git

宠物数据

    wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    tar -xvf images.tar.gz
    tar -xvf annotations.tar.gz

The Tensorflow Object Detection API expects data to be in the TFRecord format, so we'll now run the create_pet_tf_record script to convert from the raw Oxford-IIIT Pet dataset into TFRecords. Run the following commands from the tensorflow/models/research/ directory:
 
    # From tensorflow/models/research/
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --label_map_path=object_detection/data/pet_label_map.pbtxt \
        --data_dir=`pwd`/data/pets \
        --output_dir=`pwd`/training_data/pets

下载训练好的模型

    $ cd model
    $ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    $ tar -xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    
    $ cp faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.* ../training_data/pets
    
复制 pet_label_map.pbtxt 到 training_data/pets 目录下

    cp object_detection/data/pet_label_map.pbtxt training_data/pets/
    cp object_detection/samples/configs/faster_rcnn_resnet101_pets.config training_data/pets/


We'll need to configure some paths in order for the template to work. Search the file for instances of PATH_TO_BE_CONFIGURED and replace them with the appropriate value 

    vi training_data/pets/faster_rcnn_resnet101_pets.config
    
replace PATH_TO_BE_CONFIGURED to 'training_data/pets'


    python object_detection/model_main.py \
        --pipeline_config_path=training_data/pets/faster_rcnn_resnet101_pets.config \
        --model_dir=training_data/pets \
        --num_train_steps=50000 \
        --sample_1_of_n_eval_examples=1 \
        --alsologtostderr