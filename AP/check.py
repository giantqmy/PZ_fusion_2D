import os

def align_dt_with_gt(dt_dir, gt_dir):
    """
    如果 dt 中缺少 gt 中的标签文件，则在 dt 中补空文件
    """
    if not os.path.exists(dt_dir):
        raise FileNotFoundError(f"dt 标签目录不存在：{dt_dir}")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"gt 标签目录不存在：{gt_dir}")

    # 读取 gt 下所有 txt 文件
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]

    print(f"GT标签文件数量：{len(gt_files)}")

    # 统计补了多少空文件
    created = 0

    for filename in gt_files:
        dt_path = os.path.join(dt_dir, filename)

        # 如果 dt 中不存在就补空文件
        if not os.path.exists(dt_path):
            open(dt_path, 'w', encoding='utf-8').close()
            created += 1

    print(f"已补齐 {created} 个空标签文件。")
    print("操作完成！")


if __name__ == "__main__":
    dt_labels = r"/media/ddc/新加卷/hys/qmy/PZ_8/runs/predict017+RGBad+nirad+depth改了做多avgpool+split里面的interpolate负号改+fpm+c2f+去sincos+cbam+c3k2——2004/output"  # TODO: 改成你的模型推理labels路径
    gt_labels = r"/media/ddc/新加卷/hys/qmy/PZ_8/AP/valgt"  # TODO: 改成你的GT标签路径

    align_dt_with_gt(dt_labels, gt_labels)
