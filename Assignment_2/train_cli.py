import argparse
import os
from finetuning import train_test, train_test_sep_val

def main():
    parser = argparse.ArgumentParser(description="Train and test models with specified configurations.")

    # Adding arguments
    parser.add_argument('--source_langs', nargs='+', help='Source languages for training, separated by space ', required=True)
    parser.add_argument('--target_langs', nargs='+', help='Target languages for training and evaluation, separated by space ', required=True)
    parser.add_argument('--model_name', help='Model name or path, example: "/fp/projects01/ec30/models/xlm-roberta-base/"', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
    parser.add_argument('--epoch', type=int, help='Number of epochs for training', default=20)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=1e-5)
    parser.add_argument('--finetune', action='store_true', help='Flag to finetune the model. If not set, will freeze the transformer layers.')
    parser.add_argument('--separate_val', action='store_true', help='Flag to use separated validation for each language for Task 2. Also saves the plots.')

    args = parser.parse_args()
    print("args", args)

    # Automatically generate save path from model name
    model_basename = os.path.basename(args.model_name.rstrip('/'))
    save_path_folder = f"./{model_basename}_{''.join(args.source_langs)}__{''.join(args.target_langs)}/"
    os.makedirs(save_path_folder, exist_ok=True)

    #all the appropriate function based on the separate_val flag
    if args.separate_val:
        train_test_sep_val(args.source_langs, args.target_langs, model_path=args.model_name, save_path=save_path_folder,  bs=args.batch_size, lr=args.learning_rate, finetune=args.finetune, epochs=args.epoch)
    else:
        train_test(args.source_langs, args.target_langs, model_path=args.model_name, save_path=save_path_folder, bs=args.batch_size, lr=args.learning_rate, finetune=args.finetune, epochs=args.epoch)

    print(f"Training and evaluation completed. Results saved to {save_path_folder}")

if __name__ == "__main__":
    main()
