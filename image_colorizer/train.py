import torch
import os
import time
from .checkpoint import save_checkpoint

class ColorizationTrainer:
    def __init__(self, model, train_dl, val_dl, epochs=10, display_every=1, save_every=1, 
                 save_every_batch=10, use_wandb=False, patience=5, save_path="checkpoint.pth"):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.epochs = epochs
        self.display_every = display_every
        self.save_every = save_every
        self.save_every_batch = save_every_batch
        self.use_wandb = use_wandb
        self.patience = patience
        self.save_path = save_path
        
        # Scheduler for LR decay
        self.sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.model.opt_G, T_max=epochs)
        self.sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.model.opt_D, T_max=epochs)
        
        self.best_ssim = -1
        self.no_improve_epochs = 0

    def train(self):
        print(f"\n============================================================")
        print(f"  Training Image Colorization Model")
        print(f"  Epochs: {self.epochs} | Batches/epoch: {len(self.train_dl)}")
        print(f"  Device: {self.model.device}")
        print(f"  Checkpoint Frequency: {self.save_every_batch} batches")
        print(f"============================================================\n")
        
        for e in range(self.epochs):
            self.model.train()
            for i, data in enumerate(self.train_dl):
                self.model.setup_input(data)
                self.model.optimize()
                
                # Simple progress logging
                if i % self.display_every == 0:
                    print(f"Epoch {e+1}/{self.epochs} | Batch {i}/{len(self.train_dl)} | "
                          f"Loss_G: {self.model.loss_G.item():.4f} | Loss_D: {self.model.loss_D.item():.4f}")
                          
                # NEW: Batch-based checkpointing
                if i > 0 and i % self.save_every_batch == 0:
                    save_checkpoint(self.model, e + 1, self.save_path)
                    print(f"!!! Periodic Checkpoint Saved (Batch {i}) !!!")
                    
            # Step schedulers
            self.sched_G.step()
            self.sched_D.step()
            
            # Save checkpoint (End of epoch)
            if (e + 1) % self.save_every == 0:
                save_checkpoint(self.model, e + 1, self.save_path)
                print(f"Epoch {e+1} Complete. Checkpoint saved: {self.save_path}")
                
        print("\nTraining Complete!")


if __name__ == "__main__":
    import argparse
    from .dataset import ColorizationDataLoader
    from .model import MainModel
    from .checkpoint import load_checkpoint

    parser = argparse.ArgumentParser(description="Train Image Colorization Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of COCO images to use")
    parser.add_argument("--save_path", type=str, default="model_checkpoint.pth", help="Path to save checkpoints")
    parser.add_argument("--save_every_batch", type=int, default=10, help="Save checkpoint every X batches")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--lr_G", type=float, default=2e-4, help="Learning rate for Generator")
    parser.add_argument("--lr_D", type=float, default=2e-4, help="Learning rate for Discriminator")
    parser.add_argument("--use_attention", action="store_true", help="Use self-attention in generator")

    args = parser.parse_args()

    # 1. Initialize Data
    print("Loading data...")
    data_loader = ColorizationDataLoader(
        num_images=args.num_images, 
        batch_size=args.batch_size
    )
    train_dl, val_dl = data_loader.get_dataloaders()

    # 2. Initialize Model
    print("Initializing model...")
    model = MainModel(
        lr_G=args.lr_G, 
        lr_D=args.lr_D, 
        use_attention=args.use_attention
    )

    # 3. Resume if requested
    if args.resume and os.path.exists(args.save_path):
        print(f"Resuming from {args.save_path}...")
        load_checkpoint(args.save_path, model)

    # 4. Start Training
    trainer = ColorizationTrainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=args.epochs,
        save_path=args.save_path,
        save_every_batch=args.save_every_batch,
        display_every=1 # Maximum visibility
    )
    
    trainer.train()
