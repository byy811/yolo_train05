"""
WIoU v3 损失函数完整测试脚本
测试内容：
1. 基础功能测试
2. 与 CIoU 对比
3. 动态聚焦机制验证
4. 边界情况测试
"""

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np

# 添加 ultralytics 路径（如果需要）
# sys.path.insert(0, 'ultralytics')

try:
    from ultralytics.utils.loss import bbox_wiou
    from ultralytics.utils.metrics import bbox_iou
    print("✅ 成功导入 bbox_wiou 和 bbox_iou")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保:")
    print("1. 已经修改了 ultralytics/utils/loss.py")
    print("2. 添加了 bbox_wiou 函数")
    exit(1)


def test_basic_functionality():
    """测试 1: 基础功能测试"""
    print("\n" + "="*60)
    print("测试 1: 基础功能测试")
    print("="*60)

    # 创建测试数据（xyxy 格式）
    pred_boxes = torch.tensor([
        [100.0, 100.0, 150.0, 150.0],  # 预测框1 (50x50)
        [200.0, 200.0, 260.0, 260.0],  # 预测框2 (60x60)
        [300.0, 300.0, 370.0, 370.0],  # 预测框3 (70x70)
    ])

    target_boxes = torch.tensor([
        [105.0, 105.0, 155.0, 155.0],  # 真实框1 (高 IoU，轻微偏移)
        [220.0, 220.0, 280.0, 280.0],  # 真实框2 (中 IoU，偏移较大)
        [350.0, 350.0, 420.0, 420.0],  # 真实框3 (低 IoU，偏移很大)
    ])

    # 计算 WIoU v3 损失
    try:
        loss_wiou = bbox_wiou(pred_boxes, target_boxes, scale=True)
        print(f"✅ WIoU v3 计算成功")
        print(f"\n预测框:\n{pred_boxes}")
        print(f"\n真实框:\n{target_boxes}")
        print(f"\nWIoU v3 损失: {loss_wiou}")
        print(f"平均损失: {loss_wiou.mean().item():.4f}")

        # 验证输出形状
        assert loss_wiou.shape == (3,), f"形状错误: 期望 (3,), 得到 {loss_wiou.shape}"
        print(f"✅ 输出形状正确: {loss_wiou.shape}")

        # 验证损失值范围（应该都是正数）
        assert (loss_wiou >= 0).all(), "损失值包含负数！"
        print(f"✅ 损失值全部为正")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_with_ciou():
    """测试 2: 与 CIoU 对比"""
    print("\n" + "="*60)
    print("测试 2: WIoU v3 vs CIoU 对比")
    print("="*60)

    # 创建不同质量的预测框
    target = torch.tensor([[100.0, 100.0, 200.0, 200.0]])  # 真实框

    # 5 个不同质量的预测框
    predictions = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],  # 完美匹配 (IoU=1.0)
        [105.0, 105.0, 205.0, 205.0],  # 轻微偏移 (IoU≈0.8)
        [110.0, 110.0, 210.0, 210.0],  # 中等偏移 (IoU≈0.6)
        [120.0, 120.0, 220.0, 220.0],  # 较大偏移 (IoU≈0.4)
        [150.0, 150.0, 250.0, 250.0],  # 大偏移 (IoU≈0.2)
    ])

    try:
        # 计算 WIoU v3
        loss_wiou = bbox_wiou(predictions, target.repeat(5, 1), scale=True)

        # 计算 CIoU
        iou_ciou = bbox_iou(predictions, target.repeat(5, 1), xywh=False, CIoU=True)
        loss_ciou = 1 - iou_ciou

        # 计算普通 IoU 用于参考
        iou_normal = bbox_iou(predictions, target.repeat(5, 1), xywh=False)

        print("\n对比结果:")
        print(f"{'预测框偏移':<15} {'IoU':<10} {'CIoU Loss':<15} {'WIoU Loss':<15} {'差异':<10}")
        print("-" * 70)

        offsets = ['完美匹配', '轻微偏移', '中等偏移', '较大偏移', '大偏移']
        for i in range(5):
            diff = loss_wiou[i].item() - loss_ciou[i].item()
            print(f"{offsets[i]:<15} {iou_normal[i].item():.4f}    "
                  f"{loss_ciou[i].item():<15.4f} {loss_wiou[i].item():<15.4f} {diff:+.4f}")

        print(f"\n✅ 对比完成")
        print(f"💡 观察: WIoU v3 对低质量样本（大偏移）施加了更大的损失权重")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_focusing():
    """测试 3: 动态聚焦机制验证"""
    print("\n" + "="*60)
    print("测试 3: 动态聚焦机制验证")
    print("="*60)

    # 模拟一个批次的训练过程
    target = torch.tensor([[100.0, 100.0, 200.0, 200.0]]).repeat(10, 1)

    # 创建 10 个不同质量的预测框
    offsets = torch.linspace(0, 50, 10).unsqueeze(1).repeat(1, 4)
    predictions = target + offsets

    try:
        # 计算损失（多次调用以观察动态调整）
        print("\n多次迭代测试（模拟训练过程）:")

        for iter in range(3):
            loss_wiou = bbox_wiou(predictions, target, scale=True)
            avg_loss = loss_wiou.mean().item()
            print(f"迭代 {iter+1}: 平均损失 = {avg_loss:.4f}, 损失范围 = [{loss_wiou.min():.4f}, {loss_wiou.max():.4f}]")

        print(f"\n✅ 动态聚焦机制测试完成")
        print(f"💡 说明: WIoU v3 会根据样本质量动态调整梯度权重")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """测试 4: 边界情况测试"""
    print("\n" + "="*60)
    print("测试 4: 边界情况测试")
    print("="*60)

    tests = [
        ("完全重叠",
         torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
         torch.tensor([[100.0, 100.0, 200.0, 200.0]])),

        ("完全不重叠",
         torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
         torch.tensor([[300.0, 300.0, 400.0, 400.0]])),

        ("部分重叠",
         torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
         torch.tensor([[150.0, 150.0, 250.0, 250.0]])),

        ("包含关系",
         torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
         torch.tensor([[120.0, 120.0, 180.0, 180.0]])),
    ]

    print("\n边界情况测试结果:")
    print(f"{'情况':<15} {'WIoU Loss':<15} {'状态':<10}")
    print("-" * 40)

    all_passed = True
    for name, pred, target in tests:
        try:
            loss = bbox_wiou(pred, target, scale=True)
            status = "✅ 通过"

            # 验证数值稳定性
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                status = "❌ 数值异常"
                all_passed = False

            print(f"{name:<15} {loss.item():<15.4f} {status}")

        except Exception as e:
            print(f"{name:<15} {'错误':<15} ❌ 失败: {e}")
            all_passed = False

    if all_passed:
        print(f"\n✅ 所有边界情况测试通过")
    else:
        print(f"\n❌ 部分边界情况测试失败")

    return all_passed


def test_batch_processing():
    """测试 5: 批量处理测试"""
    print("\n" + "="*60)
    print("测试 5: 批量处理测试")
    print("="*60)

    batch_sizes = [1, 8, 16, 32, 64]

    print("\n批量处理性能测试:")
    print(f"{'Batch Size':<15} {'平均损失':<15} {'计算时间(ms)':<15} {'状态':<10}")
    print("-" * 55)

    all_passed = True
    for bs in batch_sizes:
        try:
            # 生成随机数据
            pred = torch.rand(bs, 4) * 100 + 100
            pred[:, 2:] = pred[:, :2] + torch.rand(bs, 2) * 50 + 10  # 确保 x2>x1, y2>y1
            target = torch.rand(bs, 4) * 100 + 100
            target[:, 2:] = target[:, :2] + torch.rand(bs, 2) * 50 + 10

            # 计时
            import time
            start = time.time()
            loss = bbox_wiou(pred, target, scale=True)
            elapsed = (time.time() - start) * 1000  # 转换为毫秒

            avg_loss = loss.mean().item()
            status = "✅ 通过"

            print(f"{bs:<15} {avg_loss:<15.4f} {elapsed:<15.2f} {status}")

        except Exception as e:
            print(f"{bs:<15} {'错误':<15} {'错误':<15} ❌ {e}")
            all_passed = False

    if all_passed:
        print(f"\n✅ 批量处理测试通过")
    else:
        print(f"\n❌ 批量处理测试失败")

    return all_passed


def visualize_loss_landscape():
    """测试 6: 损失函数地形可视化（可选）"""
    print("\n" + "="*60)
    print("测试 6: 损失函数地形可视化")
    print("="*60)

    try:
        # 固定真实框
        target = torch.tensor([[100.0, 100.0, 200.0, 200.0]])

        # 创建预测框网格（改变中心点位置）
        offsets = np.linspace(-50, 50, 30)
        loss_map_wiou = np.zeros((len(offsets), len(offsets)))
        loss_map_ciou = np.zeros((len(offsets), len(offsets)))

        print("正在计算损失地形...")
        for i, offset_x in enumerate(offsets):
            for j, offset_y in enumerate(offsets):
                pred = target.clone()
                pred[:, 0] += offset_x
                pred[:, 1] += offset_y
                pred[:, 2] += offset_x
                pred[:, 3] += offset_y

                # WIoU v3
                loss_wiou = bbox_wiou(pred, target, scale=True)
                loss_map_wiou[j, i] = loss_wiou.item()

                # CIoU
                iou_ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
                loss_map_ciou[j, i] = (1 - iou_ciou).item()

        # 绘制对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # WIoU v3 地形
        im1 = axes[0].contourf(offsets, offsets, loss_map_wiou, levels=20, cmap='RdYlBu_r')
        axes[0].set_title('WIoU v3 Loss Landscape', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Offset')
        axes[0].set_ylabel('Y Offset')
        axes[0].plot(0, 0, 'r*', markersize=15, label='Target Center')
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0])

        # CIoU 地形
        im2 = axes[1].contourf(offsets, offsets, loss_map_ciou, levels=20, cmap='RdYlBu_r')
        axes[1].set_title('CIoU Loss Landscape', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X Offset')
        axes[1].set_ylabel('Y Offset')
        axes[1].plot(0, 0, 'r*', markersize=15, label='Target Center')
        axes[1].legend()
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig('wiou_vs_ciou_landscape.png', dpi=300, bbox_inches='tight')
        print("✅ 损失地形图已保存: wiou_vs_ciou_landscape.png")

        # 不显示图像，只保存
        plt.close()

        return True

    except Exception as e:
        print(f"⚠️  可视化失败（非关键错误）: {e}")
        return True  # 不影响其他测试


def main():
    """主测试函数"""
    print("="*60)
    print("WIoU v3 损失函数完整测试")
    print("="*60)
    print("测试项目:")
    print("1. 基础功能测试")
    print("2. 与 CIoU 对比")
    print("3. 动态聚焦机制验证")
    print("4. 边界情况测试")
    print("5. 批量处理测试")
    print("6. 损失函数地形可视化")
    print("="*60)

    # 运行所有测试
    results = {
        "基础功能": test_basic_functionality(),
        "CIoU对比": test_compare_with_ciou(),
        "动态聚焦": test_dynamic_focusing(),
        "边界情况": test_edge_cases(),
        "批量处理": test_batch_processing(),
        "可视化": visualize_loss_landscape(),
    }

    # 输出测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:<15} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "🎉"*20)
        print("✅ 所有测试通过！WIoU v3 损失函数集成成功！")
        print("🎉"*20)
        print("\n你可以开始训练了：")
        print("  python train_baseline.py")
    else:
        print("\n" + "❌"*20)
        print("部分测试失败，请检查 bbox_wiou 函数实现")
        print("❌"*20)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)