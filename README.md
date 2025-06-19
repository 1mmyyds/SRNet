1.训练
  
  （1)将训练数据集和验证集放在dataset目录下的Train_data和Valid_data，训练集和验证集按照序号进行命名，例如'1.mat','2.mat'
  
   (2)将序号分别填入dataset/split_txt/Train_list和dataset/split_txt/Valid_list
   
   (3)在训练牙齿样本图像时，选用SRNet_P网络;在Dataset中的TrainDataset和ValidDataset.将mat文件中的ground_truth和rgb改为自己的变量名称
   
   (4)在训练光源样本图像时，选用SRNet_L网络;在Dataset中的TrainDataset和ValidDataset.将mat文件中的ground_truth_w和rgb_w改为自己的变量名称
   
   (5)在训练融合图像时，选用SRNet_L_fusion网络;从Dataset_fusion中导入TrainDataset和ValidDataset.同时将变量名称改为自己变量的名称
   

2.预测
 
   (1)将训练好的模型路径填入pretrained_model_path
   
   (2)将需要预测的样本序号填入Valid_list
   
   (3)在预测牙齿样本光谱时，选用SRNet_P网络;在Dataset中的TrainDataset和ValidDataset.将mat文件中的ground_truth和rgb改为自己的变量名称
   
   (4)在预测光源样本光谱时，选用SRNet_L网络;在Dataset中的TrainDataset和ValidDataset.将mat文件中的ground_truth_w和rgb_w改为自己的变量名称
   
   (5)在预测光谱反射率时，选用SRNet_L_fusion网络;从Dataset_fusion中导入TrainDataset和ValidDataset.同时将变量名称改为自己变量的名称
