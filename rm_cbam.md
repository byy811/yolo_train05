D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\.venv\Scripts\python.exe D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\trainfinal.py
======================================================================
🚀 开始消融实验: CBAM
======================================================================

⚠️  当前实验需要使用 CIoU 损失!
请确保 loss.py 中的修改如下:
   ----------------------------------------
在 BboxLoss.forward() 中:
✅ 使用: iou = bbox_iou(..., CIoU=True)
✅ 使用: loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
❌ 注释: # loss_wiou = bbox_wiou(...)
   ----------------------------------------
按回车键确认已修改loss.py...

📦 加载模型: YOLOv8n-CBAM (改进版)
WARNING no model scale passed. Assuming scale='n'.
🔄 正在加载预训练权重...
Transferred 162/358 items from pretrained weights
✅ 预训练权重加载成功

======================================================================
⏰ 开始训练...
======================================================================

New https://pypi.org/project/ultralytics/8.4.21 available  Update with 'pip install -U ultralytics'
Ultralytics 8.3.246  Python-3.10.11 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
engine\trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=traffic_sign.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8-cbam.yaml, momentum=0.937, mosaic=1.0, multi_scale=False, name=cbam, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=yolov8n.pt, profile=False, project=runs/ablation/030601, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\cbam, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
Overriding model.yaml nc=80 with nc=15
WARNING no model scale passed. Assuming scale='n'.

                   from  n    params  module                                       arguments                     
0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
10                  -1  1     65890  ultralytics.nn.modules.conv.CBAM             [256, 7]                      
11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
13                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
16                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
19                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
22                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
23        [16, 19, 22]  1    754237  ultralytics.nn.modules.head.Detect           [15, [64, 128, 256]]          
YOLOv8-cbam summary: 134 layers, 3,079,663 parameters, 3,079,647 gradients, 8.3 GFLOPs

Transferred 322/358 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed
train: Fast image access  (ping: 0.10.0 ms, read: 336.5184.2 MB/s, size: 23.3 KB)
train: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\train\labels.cache... 3530 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3530/3530  0.0s
val: Fast image access  (ping: 0.10.1 ms, read: 352.6166.3 MB/s, size: 21.4 KB)
val: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\valid\labels.cache... 801 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 801/801  0.0s
Plotting labels to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\cbam\labels.jpg...
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000526, momentum=0.9) with parameter groups 57 weight(decay=0.0), 66 weight(decay=0.0005), 64 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\cbam
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.06G      2.681      4.744      3.814         37        640: 100% ━━━━━━━━━━━━ 221/221 4.5it/s 49.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.415       0.26      0.118     0.0719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      1.99G      1.529      3.386      2.521         23        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.0s
                   all        801        944      0.208      0.413      0.212      0.159

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      2.01G      1.249      2.901      1.958         19        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.167      0.499      0.226      0.166

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      1.99G      1.111       2.61      1.708         14        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.179      0.502      0.254       0.19

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      2.01G     0.9913      2.267      1.543         24        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.309      0.485      0.306      0.232

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      1.98G     0.9276      2.051      1.456         14        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.336       0.49      0.388      0.304

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      2.01G     0.8888      1.875      1.405         13        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.405      0.501      0.475      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      1.99G     0.8656      1.739      1.375         21        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.499      0.589      0.546      0.436

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      2.01G     0.8445      1.606      1.349         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.533      0.617       0.58       0.47

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      1.98G     0.8116      1.488       1.32         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.578       0.64      0.658      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      2.01G     0.8007      1.399      1.306         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.679       0.66      0.682      0.567

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      1.99G     0.7852      1.312      1.284         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.667      0.644      0.708      0.586

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      2.01G     0.7783      1.229      1.286         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.738       0.66      0.733      0.607

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      1.98G     0.7609      1.195      1.266         18        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944       0.84       0.71      0.805      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      2.01G     0.7491      1.122      1.252         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.878      0.737      0.827      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      1.99G     0.7294      1.082      1.236         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.876      0.768      0.842      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      2.01G     0.7203      1.042      1.221         25        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.916       0.78      0.859      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      1.96G     0.7108     0.9846      1.215         13        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.897       0.77      0.865      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      1.99G     0.7035     0.9717      1.208         26        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.909      0.798      0.872      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      2.01G     0.6936     0.9435      1.207         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.918      0.789      0.871      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      1.99G     0.7018     0.9527      1.204         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.912      0.778      0.875      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100         2G     0.6802     0.8914      1.191         16        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.912      0.811       0.89       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      2.15G     0.6886     0.8913      1.191         31        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.885      0.785      0.879      0.737

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      1.99G     0.6634     0.8774      1.186         19        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.932      0.809      0.901      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      2.01G     0.6702     0.8534       1.18         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.918      0.821      0.902      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      1.98G     0.6771     0.8485      1.176         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.906      0.802      0.898      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      2.01G     0.6609     0.8301      1.166         17        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.928      0.811      0.898      0.756

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      1.97G     0.6441     0.8212      1.156         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.934      0.828      0.903      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      1.99G     0.6482      0.812      1.149         26        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.935      0.812      0.898      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100         2G     0.6489     0.7885      1.158         30        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.942      0.815       0.91      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      2.16G     0.6396     0.7834      1.151         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.933       0.84      0.912      0.773

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      1.97G     0.6351     0.7814      1.145         15        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.938      0.827      0.911      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      1.99G     0.6314     0.7568      1.142         21        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.902      0.859      0.922      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      1.99G     0.6243     0.7483      1.136         20        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.934      0.827      0.917      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      2.15G     0.6109     0.7361      1.131         28        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.909      0.826      0.905      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      1.99G     0.6149     0.7384      1.129         29        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.936      0.827      0.915      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      2.01G     0.6033     0.7243      1.123         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.0s
                   all        801        944      0.917       0.87      0.926      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      1.96G     0.6152     0.7196      1.131         21        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.943      0.846      0.936      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      1.99G     0.6019      0.705      1.117         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.943      0.853      0.931      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      2.01G     0.5955     0.6977      1.113         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.906      0.852      0.921      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      1.99G     0.5961     0.6903      1.117         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.928      0.849      0.932      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100         2G     0.5946     0.6741      1.113         19        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.93       0.86      0.934      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      2.15G     0.5871     0.6758      1.111         28        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.943      0.856      0.929      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      1.99G     0.5898     0.6957      1.112         23        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.939      0.838      0.925      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      2.01G     0.5795     0.6791      1.106         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.925      0.886      0.942      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      1.96G      0.583     0.6727      1.105         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.952      0.846      0.937      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      1.99G     0.5827     0.6635      1.106         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.945      0.854      0.938      0.799

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      2.01G     0.5748     0.6545      1.093         30        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.942      0.858      0.935        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      1.97G       0.58     0.6677      1.103         12        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.949      0.852      0.934      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      1.98G     0.5736     0.6522        1.1         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.938      0.879      0.942      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      2.01G     0.5758     0.6453      1.097         15        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944       0.93      0.889      0.941      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      1.99G     0.5643     0.6467      1.096         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.946      0.889       0.95       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      2.01G     0.5589     0.6264      1.089         28        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.943      0.897      0.948      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      1.98G     0.5679     0.6257      1.087         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.954      0.887      0.948      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      2.01G     0.5579     0.6278      1.085         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.935      0.897      0.944      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      1.99G     0.5603     0.6255       1.09         33        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.963      0.904       0.95       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      2.01G     0.5575     0.6021      1.081         20        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.969      0.889      0.952      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      1.96G      0.554     0.6197      1.079         15        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.948      0.895      0.953      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      1.99G     0.5422     0.5932      1.072         33        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.946      0.904       0.95      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      2.01G     0.5516     0.6122      1.074         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.957      0.891      0.949      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      1.99G     0.5454     0.6006      1.072         24        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.932      0.908       0.95      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100         2G     0.5443     0.5849      1.075         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.966      0.884      0.947      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      2.15G     0.5349     0.5939      1.068         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.958      0.893      0.949      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      1.97G     0.5374      0.593      1.071         19        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.964      0.893      0.952      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      1.99G     0.5269     0.5732      1.065         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.962      0.915      0.958      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100         2G     0.5381     0.5781      1.067         22        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.948      0.904      0.952       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      2.16G     0.5291      0.569      1.065         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.958      0.913      0.953      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      1.99G     0.5267     0.5593      1.058         20        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.955      0.901      0.951      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      2.01G     0.5185     0.5512      1.056         29        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.953      0.903      0.955      0.823

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      1.98G     0.5264     0.5699      1.057         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.962      0.913      0.955      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      2.01G     0.5195     0.5564      1.056         24        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.962      0.907      0.955      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      1.99G     0.5163     0.5499      1.057         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.962      0.914      0.959       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      2.01G     0.5118     0.5517      1.053         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.969      0.906       0.96      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      1.96G     0.5172     0.5543      1.056         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.967      0.905      0.958      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      1.99G     0.5152     0.5465      1.053         22        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.963        0.9      0.957      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      2.01G     0.5092      0.542      1.046         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.956      0.911      0.958      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      1.99G     0.5099     0.5368      1.054         22        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.956      0.917       0.96      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100         2G     0.5007     0.5351      1.042         31        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.973      0.914      0.959      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      2.17G     0.4964     0.5345      1.047         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.969      0.913      0.959      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      1.97G     0.5067     0.5271      1.042         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.942      0.921      0.957      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      1.99G     0.4924     0.5274       1.04         23        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944       0.97      0.904      0.961      0.828

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      1.99G     0.4953     0.5137      1.038         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.954      0.921      0.962      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      2.16G     0.4967      0.518      1.039         24        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.959      0.919      0.961      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      1.99G     0.4945     0.5227      1.037         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.973      0.905      0.961      0.828

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100         2G     0.4878     0.5139      1.034         39        640: 100% ━━━━━━━━━━━╸ 220/221 5.8it/s 44.3s<0.2s
Traceback (most recent call last):
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\trainfinal.py", line 73, in <module>
results = model.train(
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\yolo_resource\ultralytics\engine\model.py", line 773, in train
self.trainer.train()
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\yolo_resource\ultralytics\engine\trainer.py", line 243, in train
self._do_train()
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\yolo_resource\ultralytics\engine\trainer.py", line 434, in _do_train
self.scaler.scale(self.loss).backward()
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\.venv\lib\site-packages\torch\_tensor.py", line 648, in backward
torch.autograd.backward(
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\.venv\lib\site-packages\torch\autograd\__init__.py", line 353, in backward
_engine_run_backward(
File "D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\.venv\lib\site-packages\torch\autograd\graph.py", line 824, in _engine_run_backward
return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
KeyboardInterrupt

进程已结束，退出代码为 -1073741510 (0xC000013A: interrupted by Ctrl+C)
