"""
YOLOv8 消融实验完整训练脚本
支持 4 组实验：Baseline, CBAM, WIoU, CBAM+WIoU
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 实验标签：
    #   'baseline'   -> 实验1: 原版 YOLOv8n (CIoU 损失)
    #   'cbam'       -> 实验2: YOLOv8n + CBAM (CIoU 损失)
    #   'wiou'       -> 实验3: YOLOv8n + WIoU v3
    #   'final'      -> 实验4: YOLOv8n + CBAM + WIoU v3 (完整版)

    experiment_tag = 'baseline'  # ← 【每次训练前改这里！】

    experiment_result='030602'

    # 训练参数
    EPOCHS = 100                # 正式训练用 100-150，测试用 3-10
    BATCH_SIZE = 16             # 根据显存调整：16(>6GB) / 8(4-6GB) / 4(<4GB)
    DATA_PATH = 'traffic_sign.yaml'
    DEVICE = 0                  # GPU 设备号，CPU 用 'cpu'

    # ============================================

    print("="*70)
    print(f"🚀 开始消融实验: {experiment_tag.upper()}")
    print("="*70)

    # ========= 【关键】损失函数检查 =========
    if experiment_tag in ['baseline', 'cbam']:
        print("\n⚠️  当前实验需要使用 CIoU 损失!")
        print("   请确保 loss.py 中的修改如下:")
        print("   ----------------------------------------")
        print("   在 BboxLoss.forward() 中:")
        print("   ✅ 使用: iou = bbox_iou(..., CIoU=True)")
        print("   ✅ 使用: loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum")
        print("   ❌ 注释: # loss_wiou = bbox_wiou(...)")
        print("   ----------------------------------------")
        input("   按回车键确认已修改loss.py...")

    elif experiment_tag in ['wiou', 'final']:
        print("\n✅ 当前实验使用 WIoU v3 损失!")
        print("   请确保 loss.py 中的修改如下:")
        print("   ----------------------------------------")
        print("   在 BboxLoss.forward() 中:")
        print("   ❌ 注释: # iou = bbox_iou(..., CIoU=True)")
        print("   ❌ 注释: # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum")
        print("   ✅ 使用: loss_wiou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], scale=True)")
        print("   ✅ 使用: loss_iou = (loss_wiou * weight.squeeze(-1)).sum() / target_scores_sum")
        print("   ----------------------------------------")
        input("   按回车键确认已修改loss.py...")

    # 根据实验标签选择模型配置
    if experiment_tag in ['baseline', 'wiou']:
        print("\n📦 加载模型: YOLOv8n (标准版)")
        # model = YOLO('yolov8n.pt')
        model = YOLO('yolov8.yaml')
    elif experiment_tag in ['cbam', 'final']:
        print("\n📦 加载模型: YOLOv8n-CBAM (改进版)")
        model = YOLO('yolov8-cbam.yaml')
        # print("🔄 正在加载预训练权重...")
        # try:
        #     model.load('yolov8n.pt')
        #     print("✅ 预训练权重加载成功")
        # except Exception as e:
        #     print(f"⚠️  权重加载失败: {e}")

    # 开始训练
    print("\n" + "="*70)
    print("⏰ 开始训练...")
    print("="*70 + "\n")

    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        pretrained=False,     # <--- 【关键】必须加上这一行，强制从头训练
        workers=4,                      # 数据加载线程数
        project='runs/ablation/030602',        # 保存路径
        name=experiment_tag,            # 实验名称
        patience=50,                    # 早停轮数
        save=True,                      # 保存模型
        plots=True,                     # 生成图表
        device=DEVICE,

        # 优化器设置
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        # ❌ 原来是 0.5 (必须修改!)
        # ✅ 现在改为 0.0 (严禁左右翻转)
        # 包含数字和文字标志，关闭翻转fliplr=0.5,
        fliplr=0.0,
        mosaic=1.0,
    )

    # 训练完成总结
    print("\n" + "="*70)
    print(f"✅ 实验 [{experiment_tag.upper()}] 训练完成!")
    print("="*70)
    print(f"📁 结果保存: runs/ablation/030602/{experiment_tag}/")
    print(f"📊 训练曲线: runs/ablation/030602/{experiment_tag}/results.png")
    print(f"🏆 最佳模型: runs/ablation/030602/{experiment_tag}/weights/best.pt")

    try:
        print("\n📈 训练结果摘要:")
        print(f"   mAP@0.5      : {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"   mAP@0.5:0.95 : {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"   Precision    : {results.results_dict['metrics/precision(B)']:.4f}")
        print(f"   Recall       : {results.results_dict['metrics/recall(B)']:.4f}")
    except:
        print("   (请查看 results.csv 获取详细数据)")

    print("="*70)
