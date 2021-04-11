NOTE:
  PyTorch version should >= Pytorch0.4

Installation : 
   https://github.com/mastewalhabtamu/my_detectorn2/blob/master/INSTALL.md
  
Object detection:
   python detectron2-master/demo/demo_rpn.py --config-file ../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl

After the objects are detected, we get the object and pairwise data:
   python ./gendata/get_object.py
   python ./gendata/get_pairwise.py

Download model：
The trained weight and  intermediate data has been uploaded to Google Drive，you can download them from '' https://drive.google.com/file/d/11Lv4ROr-OzmCO9cbhLByK5YtzUBBASZI/view?usp=sharing ''.
After downloading, copy the  'weights'  and  'data'  folders to the root directory, and then we will have the structure as:
       msin/
	config
	data
	detectron-master
	feeders
	gendata
	graph
	model
	weights
	ensemble.py
	main.py
	readme.txt

TESTING:
To fuse the results of human stream, object stream and pairwise stream, run test firstly to generate the scores of the softmax layer.
Considering that the data of human stream (including joints and bones) is large and it is the same with 2s-AGCN, the score of human
stream is directly given in this version of our code. The data of human stream can be obtained using the code of 2s-AGCN from 
this website (https://github.com/lshiwjx/2s-AGCN).

C-Setup:
 python main.py --config ./config/ntu120_setup/test_joint.yaml
 python main.py --config ./config/ntu120_setup/test_bone.yaml
 python main.py --config ./config/ntu120_setup/test_object.yaml
 python main.py --config ./config/ntu120_setup/test_pairwise.yaml

C-Subject:
 python main.py --config ./config/ntu120_subject/test_joint.yaml
 python main.py --config ./config/ntu120_subject/test_bone.yaml
 python main.py --config ./config/ntu120_subject/test_object.yaml
 python main.py --config ./config/ntu120_subject/test_pairwise.yaml

 Then combine the generated scores with:
 python ensemble.py


Also, we provide the intermediate data (including human skeleton, object appearance, and object position) of the first 6000 video clips in the test set under C-setup protocols making it easy to run our code.
Test:
 python main.py --config ./config/ntu120_setup_tiny/test_joint.yaml
 python main.py --config ./config/ntu120_setup_tiny/test_bone.yaml
 python main.py --config ./config/ntu120_setup_tiny/test_object.yaml
 python main.py --config ./config/ntu120_setup_tiny/test_pairwise.yaml
 Then combine the generated scores with:
 python ensemble.py
