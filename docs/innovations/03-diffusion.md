# Innovation #3 — Diffusion 取代 zi2zi

## 为什么用 Diffusion 替换 zi2zi GAN

zi2zi 是一个 pix2pix 风格的条件 GAN，对汉字字形迁移效果不错，但有三个工程痛点：

1. **训练不稳定**：判别器/生成器博弈容易发生模式崩塌。
2. **风格容量有限**：增加风格 ID 数会迅速劣化收敛。
3. **可控性差**：缺少「逐步细化」机制。

**DDPM + ControlNet** 训练目标简单，可控性强：
- MSE 噪声预测，比对抗损失稳定一个量级；
- 条件分支 channel concat 锁定字形骨架；
- 推理步数可调，速度↔质量平滑权衡。

## 集成路径（不破坏现有 API）

- 本模块以 **独立 package** (`handwrite.diffusion`) 存在，未修改任何现有文件。
- 公共 API: `DiffusionEngine`、`UNet`、`NoiseScheduler`、`train_diffusion`。
- `load_weights(path)` 在权重缺失时返回 `False` 并降级为「直接返回 condition / 空白画布」。

## 当前局限

- 仅 **scaffold**：16×16、2 时间步的合成数据。
- 未训练正式权重。
- ControlNet 分支目前是简单的 channel concat。
