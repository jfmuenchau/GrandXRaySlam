from src.train import *
from src.model import *
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser(prog="Training setup for GrandXray Slam Division A Challenge")
parser.add_argument("--model", type=str, default="res18", choices=["res18", "effb0", "convnext"],
                    help="Model architecture to use")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--ada", action="store_true", help="Use adaptive data augmentation")
parser.add_argument("--batch", type=int, default=128, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--fold", type=int, default=0, help="Fold to use for training")
parser.add_argument("--workers", type=int, default=2, help="Number of workers")
parser.add_argument("--focal", action="store_true", help="Use Focal Loss instead of BCEWithLogitsLoss")
args = parser.parse_args()

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    setup_seed()
    weights = torch.tensor(get_class_weights("data/train1.csv"), dtype=torch.float32)

    model = ModelApp(
        batch_size=args.batch,
        weights=weights,
        lr=args.lr, 
        ada=args.ada, 
        model=args.model,
        num_workers=args.workers,
        focal=args.focal
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.model
    )

    checkpoint = ModelCheckpoint(
        monitor="val_auroc",
        filename="{epoch:03d}-{val_auroc:.3f}",
        mode="max",
        save_top_k=1,
        every_n_epochs=1
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_auroc", min_delta=1e-2, patience=10),
            checkpoint
        ],
        logger=logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()