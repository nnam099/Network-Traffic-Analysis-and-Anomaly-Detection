$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$DataDir = if ($env:DATA_DIR) { $env:DATA_DIR } else { "data/" }
$SaveDir = if ($env:SAVE_DIR) { $env:SAVE_DIR } else { "checkpoints/" }
$PlotDir = if ($env:PLOT_DIR) { $env:PLOT_DIR } else { "plots/" }

python train.py `
  --data_dir $DataDir `
  --save_dir $SaveDir `
  --plot_dir $PlotDir `
  --epochs 100 `
  --batch_size 512 `
  --lr 3e-4 `
  --hidden 256 `
  --patience 20 `
  @args
