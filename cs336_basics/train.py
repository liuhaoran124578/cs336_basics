import argparse
import json
import os
import pathlib
from collections.abc import Iterator

import numpy as np
import swanlab
import torch
from tqdm import tqdm


from .model import TransformerLM
from .optimizer import AdamW
from .scheduler import get_lr_cosine_schedule
from .utils import cross_entropy, gradient_clipping, load_checkpoint, save_checkpoint

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def get_memmap_dataset(path):
    """正确加载 .npy 文件为内存映射数组"""
    return np.load(path, mmap_mode="r")


def get_batch(memmap_arr, batch_size: int, context_length: int, device: str):
    """从训练集中随机采样一个批次"""
    N = len(memmap_arr)
    start_indices = np.random.randint(0, N - context_length, size=(batch_size,))
    offsets = np.arange(context_length + 1)
    all_indices = start_indices[:, None] + offsets
    tokens_np = memmap_arr[all_indices]
    x = torch.from_numpy(tokens_np[:, :-1]).to(device, dtype=torch.long, non_blocking=True)
    y = torch.from_numpy(tokens_np[:, 1:]).to(device, dtype=torch.long, non_blocking=True)
    return x, y


def val_dataloader(
    memmap_arr: np.memmap, batch_size: int, context_length: int, device: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """为验证集创建一个确定性的数据加载器 (已修正)"""
    N = len(memmap_arr)
    num_sequences = (N - 1) // context_length
    start_indices = np.arange(num_sequences) * context_length
    for i in range(0, num_sequences, batch_size):
        batch_indices = start_indices[i : i + batch_size]
        offsets = np.arange(context_length + 1)
        all_token_indices = batch_indices[:, None] + offsets
        all_tokens_np = memmap_arr[all_token_indices]
        x_np = all_tokens_np[:, :-1]
        y_np = all_tokens_np[:, 1:]
        x = torch.from_numpy(x_np).to(device, dtype=torch.long, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, dtype=torch.long, non_blocking=True)
        yield x, y


# --- 主函数 ---
def main(args):
    # 1. 初始化 swanlab
    if swanlab:
        swanlab.init(
            project=args.project_name,
            config=vars(args),  # 将所有命令行参数记录到 swanlab 配置中
        )

    # 2. 设备和精度配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # 3. 加载模型配置并实例化模型
    with open(args.model_config_path) as f:
        model_config = json.load(f)
    model = TransformerLM(**model_config, device=device, dtype=dtype).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # 编译模型以加速
    print("Compiling model...")
    model = torch.compile(model)

    # 4. 加载数据集
    train_data = get_memmap_dataset(args.train_dataset_path)
    val_data = get_memmap_dataset(args.val_dataset_path)

    # 5. 构建优化器
    optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. 计算训练总步数 (满足 total_tokens 要求)
    # total_tokens = batch_size * train_steps * context_length
    train_steps = args.total_tokens_processed // (args.per_device_train_batch_size * args.max_seq_len)
    print(f"Total training steps calculated: {train_steps}")

    warmup_iters = int(train_steps * args.warmup_ratio)
    cosine_iters = int(train_steps * args.cosine_ratio)
    print(f"Warm-up iterations (5%): {warmup_iters}")
    print(f"Cosine decay will end at iteration (90%): {cosine_iters}")

    # 7. 恢复断点 (如果需要)
    start_iter = 0
    if args.resume_from_checkpoint:
        start_iter = load_checkpoint(args.resume_from_checkpoint, model, optimizer)
        print(f"Resumed training from iteration {start_iter}")

    # 8. 训练循环
    model.train()
    for iteration in tqdm(range(start_iter, train_steps), desc="Training"):
        # 获取当前的学习率
        lr = get_lr_cosine_schedule(iteration, args.lr, args.min_lr,warmup_iters, cosine_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 训练一步
        x, y = get_batch(train_data, args.per_device_train_batch_size, args.max_seq_len, device)

        with torch.cuda.amp.autocast(enabled=(dtype == torch.bfloat16)):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # 记录训练指标
        if swanlab and (iteration % args.log_interval == 0):
            swanlab.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                    "step": iteration,
                }
            )

        # 验证
        if (iteration + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for x_val, y_val in val_dataloader(val_data, args.per_device_val_batch_size, args.max_seq_len, device):
                    with torch.cuda.amp.autocast(enabled=(dtype == torch.bfloat16)):
                        val_logits = model(x_val)
                        val_loss = cross_entropy(val_logits.reshape(-1, val_logits.shape[-1]), y_val.reshape(-1))
                    val_losses.append(val_loss.item())
                val_loss_mean = np.mean(val_losses)
                print(f"\niter {iteration + 1:06d}: VALID loss = {val_loss_mean:.4f}")
                if swanlab:
                    swanlab.log({"val_loss": val_loss_mean, "step": iteration})
            model.train()  # 切换回训练模式

        # 保存检查点
        if (iteration + 1) % args.save_interval == 0:
            os.makedirs(args.save_path, exist_ok=True)
            ckpt_name = os.path.join(args.save_path, f"iter_{iteration + 1}.pth")
            save_checkpoint(model, optimizer, iteration + 1, ckpt_name)
            print(f"Checkpoint saved to {ckpt_name}")

    if swanlab:
        swanlab.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom Transformer-based Language Model from scratch.")

    # --- 项目和路径配置 ---
    parser.add_argument("--project_name", type=str, default="cs336-assignment1", help="swanlab project name")
    parser.add_argument(
        "--model_config_path", type=str, required=True, help="Path to the JSON file containing model hyperparameters."
    )
    parser.add_argument(
        "--train_dataset_path", type=str, default=str(DATA_DIR / "train.npy"), help="Train data file path."
    )
    parser.add_argument(
        "--val_dataset_path", type=str, default=str(DATA_DIR / "val.npy"), help="Validation data file path."
    )
    parser.add_argument("--save_path", type=str, default="./checkpoints/", help="Path to save checkpoints.")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from."
    )

    # --- 训练过程配置 ---
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_val_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument(
        "--total_tokens_processed",
        type=int,
        default=327_680_000,
        help="Total number of tokens to process during training.",
    )

    # --- 日志和保存配置 ---
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Interval (in steps) for logging training metrics."
    )
    parser.add_argument("--val_interval", type=int, default=200, help="Interval (in steps) for running validation.")
    parser.add_argument("--save_interval", type=int, default=200, help="Interval (in steps) for saving checkpoints.")

    # --- 优化器和调度器配置 ---
    parser.add_argument("--lr", type=float, default=3e-4, help="Maximum learning rate (after warmup).")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate to decay to.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,  # 默认预热占总步数的 5%
        help="The percentage of total training steps to use for linear warm-up.",
    )
    parser.add_argument(
        "--cosine_ratio",
        type=float,
        default=0.90,  # 默认余弦衰减阶段占总步数的 90%
        help="The percentage of total training steps to use for cosine annealing decay. The decay ends at this point.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
