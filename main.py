"""
CLI entry point for the ImageColorization project.

Provides a command-line interface for colorizing black and white images
using a trained U-Net + PatchGAN model. Supports single-image, batch
processing, and evaluation modes.

Usage:
    python main.py --input photo.jpg --model checkpoint.pth --output colorized.jpg
    python main.py --input ./photos --model checkpoint.pth --output ./colorized --batch
    python main.py --evaluate --model checkpoint.pth
"""

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ImageColorization",
        description=(
            "Automatic Image Colorization using Deep Learning\n"
            "Colorize black & white images using a U-Net generator with PatchGAN discriminator.\n"
            "Trained on COCO dataset in LAB color space."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Colorize a single image\n"
            "  python main.py --input photo.jpg --model checkpoint.pth --output colorized.jpg\n\n"
            "  # Batch colorize a folder of images\n"
            "  python main.py --input ./photos --model checkpoint.pth --output ./colorized --batch\n\n"
            "  # Evaluate model on validation set\n"
            "  python main.py --evaluate --model checkpoint.pth\n\n"
            "  # Use CPU instead of GPU\n"
            "  python main.py --input photo.jpg --model checkpoint.pth --output out.jpg --device cpu\n"
        ),
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input image file or directory of images.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save colorized output image or directory.",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Computation device (default: cuda). Falls back to CPU if CUDA unavailable.",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Enable batch processing mode (input should be a directory).",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=256,
        help="Image processing size in pixels (default: 256).",
    )
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Run evaluation on validation dataset (computes PSNR, SSIM).",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Factor to boost color saturation (1.0 = original).",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.0,
        help="Factor to enhance sharpness (1.0 = original).",
    )
    parser.add_argument(
        "--checkpoint-info",
        type=str,
        metavar="PATH",
        help="Print metadata info about a checkpoint file and exit.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the ImageColorization CLI."""
    args = parse_args()

    # Handle checkpoint info request
    if args.checkpoint_info:
        from image_colorizer.checkpoint import get_checkpoint_info
        try:
            info = get_checkpoint_info(args.checkpoint_info)
            print(f"\nCheckpoint Info: {args.checkpoint_info}")
            for key, value in info.items():
                print(f"   {key}: {value}")
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Handle evaluation mode
    if args.evaluate:
        _run_evaluation(args)
        return

    # Validate required arguments for colorization
    if not args.input:
        print("Error: --input is required for colorization mode.", file=sys.stderr)
        print("   Use --help for usage information.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Determine processing mode
    if args.batch or os.path.isdir(args.input):
        _run_batch(args)
    else:
        _run_single(args)


def _run_single(args: argparse.Namespace) -> None:
    """Colorize a single image."""
    from image_colorizer.inference import colorize_image

    # Determine output path
    output_path = args.output
    if output_path is None:
        name, ext = os.path.splitext(args.input)
        output_path = f"{name}_colorized{ext}"

    print(f"\nImageColorization")
    print(f"{'='*40}")
    print(f"  Input:  {args.input}")
    print(f"  Model:  {args.model}")
    print(f"  Output: {output_path}")
    print(f"  Device: {args.device}")
    if args.saturation != 1.0:
        print(f"  Saturation: {args.saturation}x")
    if args.sharpen != 1.0:
        print(f"  Sharpening: {args.sharpen}x")
    print(f"{'='*40}\n")

    start_time = time.time()

    try:
        result = colorize_image(
            image_path=args.input,
            model_path=args.model,
            device=args.device,
            output_path=output_path,
            size=args.size,
            saturation_factor=args.saturation,
            sharpen_factor=args.sharpen,
        )
        elapsed = time.time() - start_time
        print(f"\nDone! ({elapsed:.2f}s)")
        print(f"   Output size: {result.size[0]}x{result.size[1]}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during colorization: {e}", file=sys.stderr)
        sys.exit(1)


def _run_batch(args: argparse.Namespace) -> None:
    """Colorize all images in a directory."""
    from image_colorizer.inference import colorize_batch

    if not os.path.isdir(args.input):
        print(f"Error: Batch mode requires a directory. Got: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(args.input, "colorized")

    print(f"\nImageColorization -- Batch Mode")
    print(f"{'='*40}")
    print(f"  Input:  {args.input}")
    print(f"  Model:  {args.model}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {args.device}")
    if args.saturation != 1.0:
        print(f"  Saturation: {args.saturation}x")
    if args.sharpen != 1.0:
        print(f"  Sharpening: {args.sharpen}x")
    print(f"{'='*40}\n")

    try:
        summary = colorize_batch(
            input_dir=args.input,
            model_path=args.model,
            output_dir=output_dir,
            device=args.device,
            size=args.size,
            saturation_factor=args.saturation,
            sharpen_factor=args.sharpen,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during batch processing: {e}", file=sys.stderr)
        sys.exit(1)


def _run_evaluation(args: argparse.Namespace) -> None:
    """Evaluate model on the validation dataset."""
    from image_colorizer.evaluate import evaluate_model, print_metrics
    from image_colorizer.dataset import ColorizationDataLoader
    from image_colorizer.checkpoint import load_model_only
    from image_colorizer.model import MainModel

    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"\nImageColorization -- Evaluation Mode")
    print(f"{'='*40}")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print(f"{'='*40}\n")

    try:
        # Load model
        model = MainModel()
        model = load_model_only(args.model, model, args.device)

        # Load validation data
        print("Loading validation dataset...")
        data_loader = ColorizationDataLoader()
        _, val_dl = data_loader.get_dataloaders()

        # Evaluate
        print("Running evaluation...")
        metrics = evaluate_model(model, val_dl, args.device)
        print_metrics(metrics)

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
