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

======================================================================
⏰ 开始训练...
======================================================================

New https://pypi.org/project/ultralytics/8.4.21 available  Update with 'pip install -U ultralytics'
Ultralytics 8.3.246  Python-3.10.11 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
engine\trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=traffic_sign.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8-cbam.yaml, momentum=0.937, mosaic=1.0, multi_scale=False, name=cbam, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=False, profile=False, project=runs/ablation/030602, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
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

Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed
train: Fast image access  (ping: 0.20.0 ms, read: 80.732.2 MB/s, size: 23.3 KB)
train: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\train\labels... 3530 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3530/3530 1.2Kit/s 3.0s
train: New cache created: D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\train\labels.cache
val: Fast image access  (ping: 0.10.0 ms, read: 25.99.2 MB/s, size: 21.4 KB)
val: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\valid\labels... 801 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 801/801 1.4Kit/s 0.6s
val: New cache created: D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\valid\labels.cache
Plotting labels to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam\labels.jpg...
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000526, momentum=0.9) with parameter groups 57 weight(decay=0.0), 66 weight(decay=0.0005), 64 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.15G      3.086      5.352      4.142         37        640: 100% ━━━━━━━━━━━━ 221/221 4.1it/s 53.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.935     0.0282     0.0204    0.00463

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      1.99G      2.564      4.531      3.211         23        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.222      0.163     0.0806     0.0427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      2.01G      1.735      3.466      2.226         19        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.166      0.222      0.135     0.0926

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      1.99G      1.389      2.921      1.795         14        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.218      0.379      0.177      0.124

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      2.01G      1.198      2.517      1.574         24        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.245      0.531      0.201      0.159

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      1.98G      1.081      2.292      1.463         14        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.214        0.5      0.224      0.177

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      2.01G      1.032       2.13      1.403         13        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.203      0.481      0.244       0.19

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      1.99G     0.9896      2.019      1.361         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.248      0.492      0.287      0.222

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      2.01G      0.945      1.954      1.319         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.276       0.49      0.306      0.244

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      1.98G     0.9156      1.845      1.295         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.354      0.454      0.339      0.265

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      2.01G     0.8899      1.766      1.264         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.338      0.485      0.407      0.329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      1.99G     0.8669      1.707      1.244         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.372      0.493      0.419      0.333

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      2.01G     0.8622      1.637      1.233         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.436      0.488      0.455      0.357

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      1.98G     0.8363      1.585      1.218         18        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.508      0.541       0.52      0.427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      2.01G     0.8288      1.523      1.211         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.574       0.52      0.575      0.478

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      1.99G     0.7972      1.458      1.182         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.576      0.622      0.629      0.525

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      2.01G     0.7934      1.407      1.177         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.753      0.565      0.666      0.553

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      1.96G     0.7823      1.356      1.167         13        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.678      0.608      0.661      0.555

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      1.99G     0.7684      1.295       1.16         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944       0.72      0.658      0.722      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      2.01G     0.7625      1.256      1.157         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.742      0.655      0.725       0.61

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      1.99G     0.7633      1.258      1.149         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.753      0.693      0.756      0.637

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100         2G     0.7479      1.186      1.148         16        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.762      0.699      0.766      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      2.15G     0.7486      1.164      1.139         31        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.749      0.694      0.747      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      1.99G     0.7309      1.139      1.134         19        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.724      0.724      0.764      0.645

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      2.01G     0.7353      1.104       1.13         17        640: 100% ━━━━━━━━━━━━ 221/221 4.5it/s 49.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.777      0.744      0.798      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      1.98G       0.73      1.091      1.122         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.788      0.759      0.802      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      2.01G     0.7144      1.048       1.11         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944       0.78      0.781      0.828      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      1.97G     0.7092      1.029      1.104         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.841      0.771      0.841      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      1.99G     0.7093      1.027      1.105         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.879      0.756      0.838      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      1.99G     0.7083     0.9924        1.1         30        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.888      0.775      0.862      0.737

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      2.16G     0.6978     0.9799      1.101         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.863      0.775      0.848      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      1.97G     0.6898     0.9646      1.096         15        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.894      0.765      0.852      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      1.99G     0.6883     0.9509      1.091         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.864      0.775      0.859      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100         2G     0.6804     0.9255       1.09         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.895      0.768      0.861      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      2.16G     0.6683     0.8967      1.078         28        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.88       0.78      0.863      0.747

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      1.99G      0.669     0.9132      1.079         29        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.904      0.791      0.881      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      2.01G     0.6575     0.8947       1.07         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.917      0.786      0.882      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      1.96G     0.6662     0.8728      1.076         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.922      0.796      0.889      0.763

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      1.99G     0.6606       0.87      1.064         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.878      0.818      0.882      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      2.01G     0.6523     0.8538      1.059         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.911      0.795      0.882      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      1.99G     0.6493     0.8416      1.063         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.946      0.779      0.892      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100         2G     0.6517     0.8215      1.061         19        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.908      0.819      0.895      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      2.15G     0.6388     0.8225      1.055         28        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.88      0.824      0.898      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      1.99G     0.6409     0.8235      1.058         23        640: 100% ━━━━━━━━━━━━ 221/221 4.5it/s 49.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.92      0.832      0.905      0.773

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      2.01G     0.6365     0.8282      1.051         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.916      0.815      0.902      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      1.96G     0.6371     0.8046      1.052         26        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.888      0.833      0.907      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      1.99G     0.6373     0.7983      1.054         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.913      0.831      0.911      0.774

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      2.01G     0.6222     0.7903      1.044         30        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.926      0.838      0.914       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      1.98G     0.6294     0.7831      1.047         12        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.909      0.837      0.905      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      1.98G     0.6166     0.7668      1.042         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.956      0.817       0.91      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      2.01G     0.6272     0.7751      1.047         15        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.954      0.817      0.912      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      1.99G     0.6108      0.763      1.042         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.955      0.821      0.913      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      2.01G     0.6048     0.7366      1.037         28        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.946      0.836      0.916      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      1.98G     0.6173     0.7491      1.038         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.939      0.834      0.918      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      2.01G     0.6067     0.7503      1.034         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.911      0.857      0.922      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      1.99G     0.6078     0.7371      1.034         33        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.932      0.852      0.924      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      2.01G     0.6099     0.7234      1.033         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.932      0.855      0.924      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      1.96G      0.594     0.7244      1.029         15        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.931      0.856      0.927      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      1.99G     0.5867     0.7043      1.024         33        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.924       0.87      0.925      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      2.01G      0.607      0.713      1.033         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.955      0.859      0.926      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      1.99G     0.5941     0.7113      1.025         24        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.945      0.858      0.926      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100         2G     0.6001     0.6945      1.028         17        640: 100% ━━━━━━━━━━━━ 221/221 4.4it/s 49.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.931      0.867      0.928      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      2.15G     0.5834      0.704       1.02         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.933      0.873      0.932      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      1.97G     0.5953     0.7001      1.028         19        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.944      0.864      0.935      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      1.99G     0.5782     0.6763      1.015         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.935      0.877      0.932      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100         2G     0.5844     0.6853      1.017         22        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.954      0.863      0.933      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      2.16G      0.587     0.6731      1.021         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.966       0.85      0.932      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      1.99G     0.5775      0.671      1.017         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.959      0.857      0.936      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      2.01G     0.5707      0.658       1.01         29        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.952      0.868      0.936      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      1.98G     0.5798     0.6801      1.015         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.959       0.86      0.936        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      2.01G     0.5719     0.6689      1.011         24        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.952      0.876      0.933        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      1.99G     0.5608     0.6458      1.008         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.1s
                   all        801        944       0.97      0.861      0.939      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      2.01G     0.5622      0.656      1.009         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.953      0.871      0.935      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      1.96G     0.5685     0.6524      1.013         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.947      0.876      0.942      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      1.99G     0.5702     0.6487      1.012         22        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.949      0.878       0.94      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      2.01G     0.5634     0.6434      1.009         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.941       0.88       0.94      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      1.99G      0.562     0.6424      1.011         22        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.953      0.865       0.94      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100         2G      0.554     0.6408      1.001         31        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.944      0.885       0.94      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      2.17G     0.5491     0.6385      1.005         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.944       0.88      0.943      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      1.97G     0.5591     0.6199      1.005         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.963      0.869      0.941      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      1.99G     0.5443     0.6215          1         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.963      0.873      0.944       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100         2G      0.546     0.6114      0.998         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.961       0.88      0.944      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      2.15G     0.5477     0.6113     0.9988         24        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.937      0.886      0.945      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      1.99G     0.5532     0.6228      1.001         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.964      0.884      0.945      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      2.01G     0.5435     0.6138     0.9962         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.953       0.89      0.946      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100      1.98G     0.5474      0.617      1.002         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.952      0.886      0.947      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      2.01G     0.5355     0.5925     0.9916         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.959      0.887      0.945      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      1.99G      0.547     0.6071      0.999         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.957       0.89      0.948      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      2.01G      0.542     0.5972     0.9934         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.961      0.889      0.947      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      1.98G      0.539     0.5962     0.9929         19        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.959      0.882      0.949      0.814
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      2.01G     0.5071     0.3802     0.9383         12        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.958      0.891      0.948      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      1.97G     0.5014     0.3784     0.9383         12        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.966      0.889      0.949      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      1.99G     0.4968     0.3696       0.93         12        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.961      0.895      0.948      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100         2G     0.4941     0.3689     0.9279         13        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.952      0.898      0.948      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      2.16G     0.4949     0.3618     0.9356         10        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.958      0.891      0.948      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      1.97G     0.4876     0.3568     0.9303         16        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.959      0.887      0.951      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      1.99G     0.4848     0.3525     0.9249         12        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.961      0.888      0.949      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100         2G     0.4804     0.3541     0.9299         10        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.954      0.892       0.95      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      2.16G     0.4758     0.3495      0.923         12        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.96      0.891      0.952      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      1.97G     0.4805      0.354     0.9239         11        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.962      0.892      0.952      0.819

100 epochs completed in 1.364 hours.
Optimizer stripped from D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam\weights\last.pt, 6.4MB
Optimizer stripped from D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam\weights\best.pt, 6.4MB

Validating D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam\weights\best.pt...
Ultralytics 8.3.246  Python-3.10.11 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
YOLOv8-cbam summary (fused): 77 layers, 3,074,463 parameters, 0 gradients, 8.2 GFLOPs
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
all        801        944       0.96      0.891      0.951       0.82
Green Light         87        122      0.884      0.672      0.803      0.476
Red Light         74        108      0.892      0.611      0.792      0.476
Speed Limit 100         52         52      0.935      0.962      0.991      0.896
Speed Limit 110         17         17      0.922          1      0.992      0.909
Speed Limit 120         60         60          1      0.983      0.995      0.916
Speed Limit 20         56         56      0.994      0.982      0.987      0.883
Speed Limit 30         71         74      0.976      0.959      0.981      0.909
Speed Limit 40         53         55          1        0.9      0.977      0.854
Speed Limit 50         68         71      0.984      0.859       0.96      0.853
Speed Limit 60         76         76      0.972      0.898      0.957      0.874
Speed Limit 70         78         78       0.99      0.962      0.985      0.906
Speed Limit 80         56         56      0.921      0.946      0.985      0.874
Speed Limit 90         38         38      0.983      0.737      0.922      0.733
Stop         81         81      0.995          1      0.995      0.918
Speed: 0.3ms preprocess, 3.1ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030602\cbam

======================================================================
✅ 实验 [CBAM] 训练完成!
======================================================================
📁 结果保存: runs/ablation/030602/cbam/
📊 训练曲线: runs/ablation/030602/cbam/results.png
🏆 最佳模型: runs/ablation/030602/cbam/weights/best.pt

📈 训练结果摘要:
mAP@0.5      : 0.9515
mAP@0.5:0.95 : 0.8198
Precision    : 0.9604
Recall       : 0.8909
======================================================================

进程已结束，退出代码为 0
