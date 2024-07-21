from train import run_pretraining
from finetune import run_finetuning
from test import run_testing

if __name__ == "__main__":
    run_pretraining()
    run_finetuning()
    run_testing()
