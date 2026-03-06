D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\.venv\Scripts\python.exe D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\trainfinal.py
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

======================================================================
⏰ 开始训练...
======================================================================

New https://pypi.org/project/ultralytics/8.4.21 available  Update with 'pip install -U ultralytics'
Ultralytics 8.3.246  Python-3.10.11 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
engine\trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=traffic_sign.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=baseline, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=runs/ablation/030601, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
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
22        [15, 18, 21]  1    754237  ultralytics.nn.modules.head.Detect           [15, [64, 128, 256]]          
Model summary: 129 layers, 3,013,773 parameters, 3,013,757 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed
train: Fast image access  (ping: 0.20.1 ms, read: 68.031.8 MB/s, size: 23.3 KB)
train: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\train\labels... 3530 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3530/3530 2.0Kit/s 1.7s
train: New cache created: D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\train\labels.cache
val: Fast image access  (ping: 0.10.0 ms, read: 25.812.1 MB/s, size: 21.4 KB)
val: Scanning D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\valid\labels... 801 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 801/801 1.4Kit/s 0.6s
val: New cache created: D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\datasets\mineset\valid\labels.cache
Plotting labels to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline\labels.jpg...
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000526, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.13G      0.835      3.465      1.187         37        640: 100% ━━━━━━━━━━━━ 221/221 3.9it/s 56.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 3.7it/s 7.0s
                   all        801        944      0.256      0.491      0.286      0.226

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      1.97G     0.7556      2.466      1.099         23        640: 100% ━━━━━━━━━━━━ 221/221 4.6it/s 48.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.5it/s 5.7s
                   all        801        944      0.465      0.469      0.504      0.413

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100         2G     0.7508       1.99      1.089         19        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.664      0.576      0.658      0.533

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      2.01G     0.7342      1.684      1.082         14        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.736      0.705      0.776      0.645

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      1.97G     0.6989      1.415       1.05         24        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.823      0.733      0.813      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      1.99G     0.6688      1.237      1.038         14        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944       0.88      0.761      0.846      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      2.01G     0.6702      1.134      1.034         13        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.828      0.777      0.839       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      1.97G     0.6598      1.071      1.033         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.929       0.79      0.895      0.752

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      1.99G     0.6448      0.994       1.02         27        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.903      0.783      0.877      0.739

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100         2G     0.6316     0.9542      1.021         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.939      0.813      0.903      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      2.08G     0.6374     0.9247      1.012         27        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.4s
                   all        801        944      0.942      0.823       0.91      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      1.97G     0.6163     0.8713      1.003         25        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.927      0.839      0.904      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      1.99G     0.6216     0.8353      1.003         25        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.926      0.786      0.888      0.744

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100         2G      0.613     0.8449          1         18        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.934      0.835      0.919      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      2.08G     0.6103     0.8088      1.001         25        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.944      0.856      0.931      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      1.97G     0.5977     0.7763     0.9936         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.926      0.861      0.933      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      1.99G     0.5958     0.7659      0.992         25        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.941      0.858      0.938       0.79

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100         2G     0.5902     0.7516     0.9861         13        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.942      0.864      0.935      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      1.96G     0.5821     0.7323     0.9883         26        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.939      0.848      0.933      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      1.98G     0.5806     0.7191     0.9872         23        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.957      0.867      0.941      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100         2G     0.5867     0.7301     0.9836         25        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.954       0.85      0.932      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100         2G     0.5775     0.6982     0.9854         16        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.923      0.877      0.945      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      1.99G     0.5776     0.7001      0.984         31        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.943      0.864      0.943      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100         2G     0.5694     0.6892     0.9822         19        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.939      0.891      0.946      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      2.02G     0.5699     0.6612     0.9777         17        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.925      0.882      0.945      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      1.96G     0.5764     0.6711     0.9797         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.962      0.869      0.949      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100         2G     0.5604     0.6479     0.9741         17        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.6it/s 5.7s
                   all        801        944      0.939      0.887      0.946      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      2.01G      0.559     0.6474     0.9704         23        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.935      0.899      0.955      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      1.98G     0.5618     0.6435     0.9709         26        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.938      0.891      0.954      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      1.98G     0.5647     0.6317     0.9726         30        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.4s
                   all        801        944      0.946      0.896      0.949      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      2.01G      0.552     0.6266     0.9678         26        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944       0.96      0.895      0.958      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      1.98G      0.545     0.6218     0.9721         15        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.966      0.885      0.954      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      1.99G     0.5448     0.6143     0.9683         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.959      0.899      0.961      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100         2G      0.544     0.5965     0.9682         20        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.964       0.89      0.953      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      2.08G     0.5281     0.5848     0.9582         28        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944       0.96      0.875      0.953      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      1.99G     0.5416     0.5897     0.9615         29        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.951      0.885      0.952      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      2.01G     0.5246     0.5929     0.9554         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.957      0.905      0.959      0.821

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      1.96G     0.5383     0.5841     0.9648         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944       0.96      0.904      0.959      0.821

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      1.99G     0.5276     0.5732     0.9538         25        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.944      0.915      0.959      0.823

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      2.01G     0.5265     0.5624     0.9542         23        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.943        0.9      0.961      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      1.97G     0.5142     0.5515     0.9522         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.939        0.9      0.961      0.821

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      1.98G     0.5276     0.5501     0.9567         19        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.934      0.917      0.958      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      2.01G     0.5171     0.5491     0.9541         28        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.955      0.926       0.96      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      1.98G      0.516     0.5557     0.9559         23        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.942      0.921      0.962      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      1.99G     0.5183     0.5584      0.951         18        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.959      0.921      0.965      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100         2G     0.5127     0.5416     0.9493         26        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.922      0.923       0.96      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      2.08G     0.5164     0.5448     0.9521         20        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.955      0.912      0.964      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      1.97G     0.5108     0.5353     0.9493         30        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.974       0.91      0.968      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100         2G     0.5059     0.5391     0.9477         12        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944       0.95      0.927      0.965      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100         2G     0.5045     0.5348      0.947         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.6it/s 5.6s
                   all        801        944      0.954      0.912      0.965      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      2.08G     0.5073     0.5241       0.95         15        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.4s
                   all        801        944      0.939      0.916      0.962      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      1.97G     0.5014      0.523     0.9512         26        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.5it/s 5.7s
                   all        801        944      0.968      0.908      0.966      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      1.99G     0.4987     0.5009     0.9479         28        640: 100% ━━━━━━━━━━━━ 221/221 4.7it/s 47.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.947      0.938      0.967      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100         2G     0.5031     0.5068     0.9419         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.4s
                   all        801        944      0.966      0.913      0.968      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      1.97G     0.4962     0.5127     0.9457         25        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.961      0.912      0.968      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      1.98G     0.4912     0.5067     0.9405         33        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.1it/s 6.3s
                   all        801        944      0.961      0.924      0.967      0.828

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100         2G     0.4963     0.4883     0.9412         20        640: 100% ━━━━━━━━━━━━ 221/221 4.7it/s 46.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.958      0.928      0.968      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      2.01G     0.4898     0.5081     0.9392         15        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.6s
                   all        801        944      0.968      0.919      0.968      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      1.98G     0.4829     0.4739     0.9368         33        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.961      0.933       0.97      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      1.98G     0.4924     0.4935     0.9423         20        640: 100% ━━━━━━━━━━━━ 221/221 4.7it/s 46.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.5it/s 5.8s
                   all        801        944      0.952      0.921      0.965       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100         2G     0.4863     0.4862      0.937         24        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.6it/s 5.7s
                   all        801        944      0.944      0.934      0.967      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100         2G     0.4869     0.4737     0.9384         17        640: 100% ━━━━━━━━━━━━ 221/221 4.7it/s 46.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.963      0.925      0.969       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      1.98G      0.477     0.4764     0.9352         21        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.5it/s 5.8s
                   all        801        944      0.971      0.912      0.968      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      1.98G     0.4847      0.481     0.9407         19        640: 100% ━━━━━━━━━━━━ 221/221 4.7it/s 46.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.6it/s 5.6s
                   all        801        944      0.967      0.919       0.97      0.837

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100         2G     0.4698     0.4682     0.9312         23        640: 100% ━━━━━━━━━━━━ 221/221 4.7it/s 46.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.955      0.932      0.971      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100         2G      0.477     0.4623     0.9344         22        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.944      0.936      0.968      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      1.98G     0.4701     0.4656     0.9352         17        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.943      0.943      0.969      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      1.98G     0.4676     0.4541     0.9319         20        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.6it/s 5.7s
                   all        801        944      0.944      0.945      0.971      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100         2G     0.4676     0.4439     0.9301         29        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.961      0.929       0.97      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100         2G     0.4694     0.4609     0.9317         25        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.962      0.922      0.968      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      1.98G      0.467     0.4506     0.9303         24        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.6s
                   all        801        944      0.953       0.93      0.969      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      1.98G     0.4575     0.4442     0.9288         18        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.956      0.939      0.972       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100         2G     0.4524     0.4458     0.9257         27        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.959       0.94      0.972      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100         2G     0.4619     0.4518     0.9314         27        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.953      0.935      0.971      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100         2G       0.46     0.4434     0.9282         22        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.4s
                   all        801        944       0.96      0.933      0.969      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100         2G     0.4555      0.441     0.9258         21        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.6s
                   all        801        944       0.95      0.937      0.969      0.833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      2.02G     0.4564     0.4362     0.9297         22        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.951      0.944       0.97      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100      1.97G      0.446     0.4279     0.9203         31        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.974      0.929      0.972      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      1.99G     0.4461     0.4269     0.9261         17        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.953      0.942      0.972       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      2.01G     0.4524     0.4223      0.925         23        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.963      0.939      0.971      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      1.97G     0.4396     0.4215     0.9217         23        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.966      0.945      0.973      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      1.98G     0.4423     0.4176     0.9172         26        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.6it/s 5.6s
                   all        801        944      0.969      0.932      0.973      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      2.01G     0.4452     0.4139     0.9231         24        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 45.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.964      0.941      0.971      0.837

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      1.98G     0.4417     0.4171     0.9188         25        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944       0.97      0.926       0.97      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      1.99G     0.4319        0.4     0.9171         27        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.951      0.949      0.972      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100         2G     0.4401     0.4185     0.9244         20        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.955      0.941      0.971      0.841

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      2.08G     0.4301     0.3966     0.9151         18        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.959      0.934       0.97      0.841

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      1.97G     0.4378      0.405     0.9178         21        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.971      0.936      0.971      0.841

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      1.99G     0.4337     0.4011     0.9165         27        640: 100% ━━━━━━━━━━━━ 221/221 4.8it/s 46.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.971      0.933      0.973      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100         2G     0.4293     0.3922     0.9141         19        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.942      0.956      0.973      0.843
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      2.08G     0.4338     0.2625     0.8775         12        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.5s
                   all        801        944      0.944      0.952      0.974      0.842

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      1.96G     0.4209     0.2565     0.8759         12        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.957      0.942      0.972      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      1.98G     0.4174     0.2547     0.8713         12        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 43.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.957      0.947      0.973      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      1.98G     0.4191     0.2517     0.8683         13        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944       0.96      0.939      0.972      0.841

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      2.01G     0.4136     0.2503      0.872         10        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.9it/s 5.3s
                   all        801        944      0.967      0.931      0.973      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      1.98G     0.4139     0.2447     0.8693         16        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.5s
                   all        801        944      0.952      0.943      0.973      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      1.99G      0.405     0.2423     0.8645         12        640: 100% ━━━━━━━━━━━━ 221/221 5.0it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.4it/s 5.9s
                   all        801        944      0.961      0.934      0.972       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100         2G     0.4095      0.242     0.8704         10        640: 100% ━━━━━━━━━━━━ 221/221 4.4it/s 50.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.8it/s 5.4s
                   all        801        944      0.966      0.935      0.973      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      1.97G     0.4028     0.2427     0.8652         12        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.0it/s 5.2s
                   all        801        944      0.968      0.932      0.973      0.843

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      1.98G     0.3999     0.2407      0.864         11        640: 100% ━━━━━━━━━━━━ 221/221 4.9it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 5.1it/s 5.1s
                   all        801        944      0.963      0.939      0.974      0.845

100 epochs completed in 1.438 hours.
Optimizer stripped from D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline\weights\last.pt, 6.3MB
Optimizer stripped from D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline\weights\best.pt, 6.3MB

Validating D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline\weights\best.pt...
Ultralytics 8.3.246  Python-3.10.11 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4096MiB)
Model summary (fused): 72 layers, 3,008,573 parameters, 0 gradients, 8.1 GFLOPs
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 4.7it/s 5.6s
all        801        944      0.963      0.939      0.974      0.845
Green Light         87        122      0.911      0.836      0.898      0.558
Red Light         74        108      0.863       0.75      0.853      0.537
Speed Limit 100         52         52      0.959          1      0.995      0.887
Speed Limit 110         17         17          1      0.923      0.984      0.916
Speed Limit 120         60         60      0.991      0.983      0.991      0.921
Speed Limit 20         56         56      0.962      0.982      0.987      0.868
Speed Limit 30         71         74       0.96      0.973      0.993      0.921
Speed Limit 40         53         55      0.962      0.982      0.993      0.882
Speed Limit 50         68         71      0.985      0.948      0.988      0.869
Speed Limit 60         76         76      0.986      0.961      0.983      0.897
Speed Limit 70         78         78       0.95      0.972      0.992       0.92
Speed Limit 80         56         56      0.972          1      0.995       0.88
Speed Limit 90         38         38          1      0.862      0.991      0.822
Stop         81         81      0.988      0.977      0.994      0.949
Speed: 0.3ms preprocess, 2.9ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to D:\c_d\d_app\d_cod_tool\jetbrain_fil\pycharm_fil\26\0306\yolo_train05\runs\ablation\030601\baseline

======================================================================
✅ 实验 [BASELINE] 训练完成!
======================================================================
📁 结果保存: runs/ablation/030601/baseline/
📊 训练曲线: runs/ablation/030601/baseline/results.png
🏆 最佳模型: runs/ablation/030601/baseline/weights/best.pt

📈 训练结果摘要:
mAP@0.5      : 0.9742
mAP@0.5:0.95 : 0.8448
Precision    : 0.9635
Recall       : 0.9392
======================================================================

进程已结束，退出代码为 0
