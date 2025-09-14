from src.train import *
from src.model import *
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from argparse import ArgumentParser


parser = ArgumentParser(prog="Training setup for GrandXray Slam Division A Challenge")
parser.add_argument("--model", type=str, default="res18", choices=["res18", "res50", "vit_b", "vit_t"],
                    help="Model architecture to use")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--ada", action="store_true", help="Use adaptive data augmentation")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--fold", type=int, default=0, help="Fold to use for training")
args = parser.parse_args()

def main():
    # Initialize model
    model = ModelApp(batch_size=parser.batch, lr=parser.lr, ada=parser.ada, model=parser.model)

    # TODO: You need to define a DataModule or dataloaders here
    # Example (if you have a DataModule class defined in src.train):
    # datamodule = GrandXrayDataModule(batch_size=args.batch, augment=args.ada)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=10)
        ]
    )

    # trainer.fit(model, datamodule=datamodule)  # Use this once datamodule is ready
    trainer.fit(model)


if __name__ == "__main__":
    main()