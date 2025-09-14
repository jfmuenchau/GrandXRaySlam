from src.train import *
from src.model import *
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
    model = ModelApp(batch_size=args.batch, lr=args.lr, ada=args.ada, model=args.model)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.model
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:03d}-{val_loss:.3f}",
        mode="min",
        save_top_k=1,
        every_n_epochs=1
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=10),
            checkpoint
        ],
        logger=logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()