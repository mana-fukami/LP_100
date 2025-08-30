import torch
import optuna
import subprocess
import sys
import os

def objective(trial):
    """Optunaが呼び出す目的関数"""
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [4, 8]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']), 
    }

    # --- worker.pyの絶対パスを取得 ---
    # __file__ は現在実行中のスクリプトのパスを指す
    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script_path = os.path.join(current_dir, "worker.py")
    
    # torchrunコマンドの構築
    world_size = torch.cuda.device_count()
    command=[
        sys.executable,
        "-u",
        "-m", "torch.distributed.run",
        "--nproc_per_node", str(world_size),
        worker_script_path,
        "--lr", str(params["learning_rate"]),
        "--batch_size", str(params["batch_size"]),
        "--optimizer", str(params["optimizer"])
    ]

    print(f"\n--- Starting Trial {trial.number} with command: {' '.join(command)} ---")

    # サブプロセスを実行
    process=subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, encoding='utf-8', bufsize=1
    )

    stdout_output = ""
    for char in iter(lambda: process.stdout.read(1), ''):
        # 1. リアルタイムでコンソールに進捗バーなどを表示
        print(char, end='', flush=True)
        # 2. 後でスコアを読み取るために、出力内容をリストに保存
        stdout_output += char

    # プロセスの終了を待つ
    process.wait()

    # エラー出力を全て読み込む
    stderr_output = process.stderr.read()

    if process.returncode != 0:
        # 異常終了した場合
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"TRIAL {trial.number} FAILED: Subprocess (worker.py) returned a non-zero exit status.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("\n--- Command that failed ---")
        print(' '.join(command))
        print("\n--- Stderr from worker.py (The Real Error) ---")
        print(stderr_output)
        raise optuna.exceptions.TrialPruned()
    else:
        # 正常終了した場合、保存したstdoutの最後の行からスコアを取得
        # worker.pyの最後のprintがスコアであることを想定
        try:
            last_line = stdout_output.strip().split('\n')[-1]
            bleu_score = float(last_line.strip())
            return bleu_score
        except (IndexError, ValueError) as e:
            print(f"Error: Could not parse score from the last line of stdout. Error: {e}")
            print("Full stdout:", stdout_output)
            raise optuna.exceptions.TrialPruned()

if __name__ == '__main__':
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("This script requires multiple GPUs to run with DDP.")
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        print("\n--- Optimization Finished ---")
        print("Number of finished trials: ", len(study.trials))
        
        best_trial = study.best_trial
        print("Best trial:")
        print(f"  Value (BLEU Score): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")