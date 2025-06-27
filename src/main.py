import argparse

from train import train
from evaluate import eval_one_image, eval_full_test
from dataloaders.dataset_kolektor import prepare_dataset as prepare_kolektor
from dataloaders.dataset_gc10 import prepare_dataset as prepare_gc10
from dataloaders.dataset_chest import prepare_dataset as prepare_chest
from dataloaders.dataset_severstal import prepare_dataset as prepare_severstal


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate SAM2 model.")
    parser.add_argument("mode", choices=["train", "eval_one_image", "eval_full_test"],
                        help="Choose operation mode")
    parser.add_argument("--dataset", choices=["kolektor", "gc10", "chest", "severstal", "severstal200", "severstal25", "severstal50", "severstal100", "severstal500"],
                        required=True,
                        help="Select dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to base SAM2 checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--fine_tuned_checkpoint", type=str,
                        help="Path to fine-tuned model (required for evaluation)")
    parser.add_argument("--model_name", type=str, default="fine_tuned_sam2",
                        help="Name for saving fine-tuned model")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--annotation_format", choices=["image", "rle", "rectangle", "bitmap"], default=None,
                        help="Format of annotations: image, rle, rectangle or bitmap")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine annotation format based on dataset if not explicitly set
    if args.annotation_format is None:
        if args.dataset == "gc10" or args.dataset == "chest":
            args.annotation_format = "image"
        elif args.dataset == "kolektor":
            args.annotation_format = "rle"

    # Prepare dataset
    if args.dataset == "kolektor":
        train_data, test_data = prepare_kolektor(data_dir="datasets/kolektorSDD2")
    elif args.dataset == "gc10":
        train_data, test_data = prepare_gc10(data_dir="datasets/gc10-DET")
    elif args.dataset == "chest":
        train_data, test_data = prepare_chest(data_dir="datasets/chest")
    elif args.dataset == "severstal":
        from dataloaders.dataset_severstal import prepare_custom_split
        train_data, test_data = prepare_custom_split(
            train_dir="datasets/severstal/train_split",
            test_dir="datasets/severstal/test_split"
        )
    elif args.dataset.startswith("severstal") and args.dataset != "severstal":
        from dataloaders.dataset_severstal import prepare_subset
        train_data, test_data = prepare_subset(f"train_{args.dataset.split('severstal')[-1]}")

    # Execute selected mode
    if args.mode == "train":
        train(args, train_data)
    elif args.mode == "eval_one_image":
        eval_one_image(args, test_data)
    elif args.mode == "eval_full_test":
        eval_full_test(args, test_data)

if __name__ == "__main__":
    main()
