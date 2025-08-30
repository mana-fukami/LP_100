import optuna
import subprocess
import sys

def objective(trial):
    """Optunaが呼び出す目的関数"""
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [4, 8]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']), 
    }
    
    # torchrunコマンドの構築
    world_size = torch.cuda.device_count()
    command=[
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node", str(world_size),
        "worker.py",
        "--lr", str(params["learning_rate"]),
        "--batch_size", str(params["batch_size"]),
        "--optimizer", str(params["optimizer"])
    ]

    print(f"\n--- Starting Trial {trial.number} with command: {' '.join(command)} ---")

    # サブプロセスを実行
    try:
        result=subprocess.run(command,check=True, capture_output=True, text=True)
        # check=True: エラーが発生したら例外を投げる
        # capture_output=True: 標準出力を取得
        # text=True: 出力を文字列として扱う

        bleu_score = float(result.stdout.strip().split("\n")[-1])
        return bleu_score
    except subprocess.CalledProcessError as e:
        # 学習が失敗した場合
        print(f"Trial {trial.number} FAILED. Stderr:\n{e.stderr}")
        # Optunaに失敗を伝え、この試行を枝刈り(prune)させる
        raise optuna.exceptions.TrialPruned()

if __name__ == '__main__':
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("This script requires multiple GPUs to run with DDP.")
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        print("\n--- Optimization Finished ---")
        print("Number of finished trials: ", len(study.trials))
        
        best_trial = study.best_trial
        print("Best trial:")
        print(f"  Value (BLEU Score): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")