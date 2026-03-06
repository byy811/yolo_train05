D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\.venv\Scripts\python.exe D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\trainfinal.py
======================================================================
🚀 开始消融实验: BASELINE
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

📦 加载模型: YOLOv8n (标准版)
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2MB 6.5MB/s 1.0s

======================================================================
⏰ 开始训练...
======================================================================

Ultralytics 8.4.21  Python-3.11.9 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
engine\trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=traffic_sign.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=baseline, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=runs/ablation/0306, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
Overriding model.yaml nc=80 with nc=15

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
10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
22        [15, 18, 21]  1    754237  ultralytics.nn.modules.head.Detect           [15, 16, None, [64, 128, 256]]
Model summary: 130 layers, 3,013,773 parameters, 3,013,757 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 7.5MB/s 0.7s
AMP: checks passed
train: Fast image access  (ping: 0.20.0 ms, read: 46.594.7 MB/s, size: 23.3 KB)
train: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\datasets\mineset\train\labels... 3530 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3530/3530 1.1Kit/s 3.3s
train: New cache created: D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\datasets\mineset\train\labels.cache
val: Fast image access  (ping: 0.40.3 ms, read: 2.61.2 MB/s, size: 21.4 KB)
val: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\datasets\mineset\valid\labels... 801 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 801/801 745.0it/s 1.1s
val: New cache created: D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\datasets\mineset\valid\labels.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000526, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Plotting labels to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline\labels.jpg...
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.04G     0.8483      3.457      1.179         37        640: 100% ━━━━━━━━━━━━ 221/221 4.4it/s 49.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.272      0.471      0.313      0.254

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      1.93G     0.7713      2.465      1.093         23        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.579      0.519      0.532      0.441

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      1.94G     0.7609      1.988      1.081         19        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.65      0.644      0.689      0.548

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      1.94G     0.7528      1.686      1.082         14        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.5it/s 5.7s
                   all        801        944       0.74      0.696      0.772      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      1.94G     0.7166      1.407       1.05         24        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.841      0.734      0.823      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      1.94G     0.6824      1.225      1.038         14        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944       0.91      0.766      0.862      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      1.94G      0.684      1.133      1.031         13        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944        0.9      0.797      0.892      0.747

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      1.94G     0.6771      1.066      1.033         21        640: 100% ━━━━━━━━━━━━ 221/221 4.5it/s 49.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 3.9it/s 6.6s
                   all        801        944      0.909      0.801      0.901      0.739

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      1.94G     0.6646     0.9915      1.022         27        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.877      0.759      0.867      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      1.94G      0.649     0.9519      1.021         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.924      0.827      0.923      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      1.96G     0.6447     0.9094      1.009         27        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.932      0.851      0.924      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      1.97G     0.6367     0.8783      1.007         25        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944       0.93       0.84       0.92       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      1.97G     0.6379     0.8281      1.003         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.928      0.847      0.932      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      1.97G     0.6231     0.8297          1         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.956      0.844      0.935      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      1.97G     0.6183     0.7919     0.9964         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.927       0.87      0.938      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      1.97G     0.6056     0.7736      0.993         21        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.939      0.882      0.947      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      1.97G      0.612     0.7664     0.9915         25        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.922      0.879      0.947      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      1.97G     0.6009     0.7325     0.9852         13        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.949      0.897      0.957        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      1.97G     0.5966     0.7409     0.9903         26        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.924      0.881      0.948      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      1.97G     0.5946     0.7303     0.9871         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.962      0.854      0.947      0.799

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      1.97G     0.5973     0.7341     0.9844         25        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.943      0.895      0.956      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100      1.97G     0.5845     0.6893      0.981         16        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.944       0.91      0.961      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      1.97G      0.588     0.6896     0.9856         31        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.947      0.896      0.959      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      1.97G     0.5781     0.6849     0.9786         19        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.953      0.912      0.959      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      1.97G     0.5872     0.6677     0.9814         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.92      0.906       0.95      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      1.97G     0.5923     0.6702     0.9784         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.946      0.913      0.962      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      1.97G     0.5773     0.6458     0.9741         17        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.958      0.896      0.959       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      1.97G     0.5692     0.6457     0.9716         23        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.95       0.92      0.965       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      1.97G     0.5709     0.6449     0.9689         26        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.941      0.919      0.962      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      1.97G     0.5772     0.6321     0.9735         30        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.954       0.92      0.967       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      1.97G     0.5724       0.63     0.9729         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.962       0.92      0.967      0.823

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      1.97G     0.5566     0.6243     0.9712         15        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.942      0.912      0.965      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      1.97G     0.5623     0.6101      0.969         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.952      0.937       0.97      0.823

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      1.97G     0.5565     0.6062     0.9674         20        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.942      0.944      0.972      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      1.97G     0.5476     0.5845     0.9596         28        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.95      0.929      0.972      0.828

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      1.97G     0.5475     0.5874     0.9595         29        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.928      0.934      0.966      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      1.97G     0.5388     0.5863     0.9557         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.946      0.939      0.969       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      1.97G     0.5491     0.5813     0.9649         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.952      0.948      0.974      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      1.97G     0.5407     0.5716     0.9537         25        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.943       0.93      0.971      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      1.97G     0.5369      0.564     0.9546         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.942      0.939      0.972      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      1.97G     0.5298     0.5626     0.9543         21        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 5.0s
                   all        801        944       0.96      0.934      0.971      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      1.97G     0.5384     0.5484     0.9547         19        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.963      0.929      0.969      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      1.97G     0.5331     0.5535     0.9553         28        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.951      0.944      0.965      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      1.97G     0.5319     0.5618     0.9568         23        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.939      0.937      0.971      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      1.97G     0.5286     0.5497     0.9506         18        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.966      0.949      0.976      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      1.97G     0.5277     0.5429     0.9514         26        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.957      0.926      0.971      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      1.97G     0.5277     0.5419     0.9506         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.959       0.95      0.973      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      1.97G     0.5244     0.5382     0.9481         30        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.959      0.938      0.969      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      1.97G     0.5215     0.5339     0.9488         12        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.961      0.946      0.971      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      1.97G     0.5159     0.5235     0.9466         21        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.956       0.93       0.97       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      1.97G     0.5217     0.5249     0.9502         15        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.946      0.948      0.972      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      1.97G     0.5106     0.5249     0.9512         26        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.963      0.939      0.971      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      1.97G     0.5071     0.5055     0.9466         28        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.948      0.953       0.97      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      1.97G     0.5135     0.5052     0.9426         21        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.953      0.938      0.972      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      1.97G     0.5094     0.5066     0.9451         25        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.949      0.952      0.972      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      1.97G     0.5038     0.5066     0.9395         33        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.963      0.946      0.973      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      1.97G     0.5087     0.4928      0.941         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944       0.95       0.95      0.971       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      1.97G     0.5003     0.5095     0.9409         15        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.941      0.953      0.976      0.837

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      1.97G     0.4975     0.4851     0.9386         33        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.955      0.949      0.973      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      1.97G     0.5068     0.4921      0.942         20        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.956      0.946      0.972      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      1.97G     0.4988     0.4877     0.9364         24        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.941      0.958      0.973      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      1.97G     0.4992     0.4767     0.9385         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944       0.95      0.946      0.971      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      1.97G     0.4876     0.4835     0.9355         21        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.95      0.954      0.973      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      1.97G     0.4954      0.487     0.9386         19        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.963      0.952      0.974      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      1.97G     0.4837     0.4698     0.9301         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.959      0.944      0.972      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100      1.97G     0.4943     0.4651     0.9344         22        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.954      0.953      0.975      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      1.97G     0.4862      0.465     0.9352         17        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.2it/s 5.0s
                   all        801        944      0.966      0.954      0.975      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      1.97G     0.4804     0.4527     0.9313         20        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.8s
                   all        801        944      0.965       0.94      0.974      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      1.97G     0.4745     0.4453     0.9284         29        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.961      0.946      0.974      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      1.97G     0.4814     0.4598     0.9302         25        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.962      0.949      0.973      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      1.97G     0.4776     0.4541     0.9282         24        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.956      0.954      0.975      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      1.97G      0.469     0.4455     0.9278         18        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944       0.96      0.943      0.975      0.841

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      1.97G     0.4636     0.4454     0.9241         27        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.952      0.961      0.975      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      1.97G     0.4744     0.4496     0.9294         27        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.955      0.956      0.974      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      1.97G     0.4737     0.4387      0.928         22        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.956      0.951      0.975      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      1.97G     0.4648     0.4415     0.9246         21        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.944      0.957      0.973       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      1.97G     0.4732     0.4375     0.9286         22        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.9s
                   all        801        944      0.959      0.949      0.977      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100      1.97G     0.4573     0.4286     0.9192         31        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.3it/s 4.9s
                   all        801        944      0.952      0.956      0.975      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      1.97G     0.4619     0.4372     0.9259         17        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.8s
                   all        801        944      0.956      0.952      0.975      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      1.97G     0.4655     0.4272     0.9234         23        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.963      0.936      0.973      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      1.97G     0.4539     0.4275     0.9202         23        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 43.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.965      0.946      0.975      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      1.97G     0.4554     0.4189     0.9171         26        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.8s
                   all        801        944       0.95      0.952      0.976      0.847

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      1.97G     0.4563     0.4179     0.9213         24        640: 100% ━━━━━━━━━━━━ 221/221 5.1it/s 42.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944       0.96      0.935       0.97      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      1.97G     0.4544     0.4193     0.9183         25        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.966      0.942      0.976      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      1.97G     0.4446      0.408     0.9164         27        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.966      0.936      0.976      0.845

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100      1.97G     0.4518     0.4193      0.922         20        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944       0.97      0.946      0.978      0.848

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      1.97G     0.4481     0.4014     0.9136         18        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.947      0.956      0.976      0.848

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      1.97G     0.4471     0.4087     0.9156         21        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.951      0.955      0.977      0.848

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      1.97G      0.448     0.4044     0.9172         27        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.4it/s 4.8s
                   all        801        944      0.954      0.957      0.978      0.851

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      1.97G     0.4415     0.3957      0.914         19        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.8s
                   all        801        944      0.946      0.948      0.976      0.846
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      1.97G     0.4483     0.2671     0.8795         12        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.952      0.939      0.974      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      1.97G     0.4378     0.2629     0.8751         12        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944       0.95       0.95      0.976      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      1.97G     0.4327      0.258     0.8691         12        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.963      0.943      0.977      0.845

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      1.97G     0.4337     0.2596     0.8677         13        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.965      0.943      0.977      0.846

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      1.97G     0.4299     0.2563      0.872         10        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.8s
                   all        801        944      0.963      0.955      0.977       0.85

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      1.97G     0.4291     0.2506     0.8698         16        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.961      0.942      0.974      0.847

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      1.97G     0.4232     0.2475     0.8629         12        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.959      0.942      0.974      0.849

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100      1.97G     0.4238     0.2476     0.8682         10        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.965      0.941      0.974      0.847

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      1.97G     0.4168     0.2466      0.864         12        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.8s
                   all        801        944      0.964       0.94      0.973      0.846

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      1.97G     0.4169      0.247     0.8636         11        640: 100% ━━━━━━━━━━━━ 221/221 5.2it/s 42.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.5it/s 4.7s
                   all        801        944      0.961      0.941      0.974      0.848

100 epochs completed in 1.344 hours.
Optimizer stripped from D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline\weights\last.pt, 6.3MB
Optimizer stripped from D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline\weights\best.pt, 6.3MB

Validating D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline\weights\best.pt...
Ultralytics 8.4.21  Python-3.11.9 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
Model summary (fused): 73 layers, 3,008,573 parameters, 0 gradients, 8.1 GFLOPs
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.4s
all        801        944      0.954      0.957      0.978       0.85
Green Light         87        122      0.881      0.869      0.926      0.554
Red Light         74        108       0.87      0.806       0.86       0.54
Speed Limit 100         52         52       0.92          1      0.994      0.906
Speed Limit 110         17         17          1      0.939      0.995      0.913
Speed Limit 120         60         60          1      0.981      0.995      0.913
Speed Limit 20         56         56      0.957      0.982      0.987      0.879
Speed Limit 30         71         74      0.953      0.986      0.991      0.933
Speed Limit 40         53         55      0.942      0.982      0.991      0.892
Speed Limit 50         68         71      0.958      0.986      0.993      0.883
Speed Limit 60         76         76      0.973      0.949      0.982      0.916
Speed Limit 70         78         78      0.977      0.987      0.995      0.913
Speed Limit 80         56         56       0.98          1      0.995      0.888
Speed Limit 90         38         38      0.968      0.947      0.991      0.831
Stop         81         81       0.98      0.988      0.992      0.945
Speed: 0.3ms preprocess, 3.0ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train\runs\detect\runs\ablation\0306\baseline

======================================================================
✅ 实验 [BASELINE] 训练完成!
======================================================================
📁 结果保存: runs/ablation/0306/baseline/
📊 训练曲线: runs/ablation/0306/baseline/results.png
🏆 最佳模型: runs/ablation/0306/baseline/weights/best.pt

📈 训练结果摘要:
mAP@0.5      : 0.9776
mAP@0.5:0.95 : 0.8504
Precision    : 0.9542
Recall       : 0.9572
======================================================================

进程已结束，退出代码为 0
