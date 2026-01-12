"""
工具函数模块
包含模型保存等通用功能
"""
import torch
from pathlib import Path


def save_models(baseline_model, dann_model, finetune_model, save_dir):
    """保存训练好的模型"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("保存模型")
    print("="*60)
    
    saved_count = 0
    
    # 保存Baseline模型
    if baseline_model is not None:
        baseline_path = save_dir / "baseline_model.pth"
        torch.save(baseline_model.state_dict(), baseline_path)
        print(f"[OK] Baseline模型已保存: {baseline_path}")
        saved_count += 1
    
    # 保存DANN模型
    if dann_model is not None:
        dann_path = save_dir / "dann_model.pth"
        torch.save(dann_model.state_dict(), dann_path)
        print(f"[OK] DANN模型已保存: {dann_path}")
        saved_count += 1
    
    # 保存FineTune模型
    if finetune_model is not None:
        finetune_path = save_dir / "finetune_model.pth"
        torch.save(finetune_model.state_dict(), finetune_path)
        print(f"[OK] FineTune模型已保存: {finetune_path}")
        saved_count += 1
    
    if saved_count == 0:
        print("[WARN] 没有模型需要保存")
    else:
        print(f"[OK] 共保存了 {saved_count} 个模型")