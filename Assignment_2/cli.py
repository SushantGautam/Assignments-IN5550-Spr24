import argparse
import os
from finetuning import train_test, train_test_sep_val

def main():
    parser = argparse.ArgumentParser(description="Train and test models with specified configurations.")

    # Adding arguments
    parser.add_argument('--source_langs', nargs='+', help='Source languages for training', required=True)
    parser.add_argument('--target_langs', nargs='+', help='Target languages for training and evaluation', required=True)
    parser.add_argument('--model_name', help='Model name or path', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=None)
    parser.add_argument('--finetune', action='store_true', help='Flag to finetune the model. If not set, will freeze the transformer layers.')

    args = parser.parse_args()

    # Automatically generate save path from model name
    model_basename = os.path.basename(args.model_name)
    save_path_folder = f"./{model_basename.replace('/', '-')}-separated" if args.separate_val else f"./{model_basename.replace('/', '-')}"
    os.makedirs(save_path_folder, exist_ok=True)

    train_test(args.source_langs, args.target_langs, model_path=args.model_name, save_path=save_path_folder, bs=args.batch_size, lr=args.learning_rate, finetune=args.finetune)

    print(f"Training and evaluation completed. Results saved to {save_path_folder}")

if __name__ == "__main__":
    main()
