import wandb
import uuid
import time
import os
from dotenv import load_dotenv

class WandBLogger:
    def __init__(self, args=None):
        self.enabled = args.use_wandb
        self.args = args
        
        if self.enabled and not args.evaluate:
            load_dotenv()
            augmentations = []
            # by default, horizontal flips and translations are always applied
            # if args.random_erase: augmentations.append("erase")
            # if args.color_jitter: augmentations.append("jitter")
            # if args.color_aug: augmentations.append("color")
            # aug_name = "+".join(augmentations) if augmentations else "base"
            #
            # aug_name = self._get_aug_name()
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project="AML-Coursework",
                name=f"{args.arch}_{args.lr:.0e}_{args.train_batch_size}_{args.optim}_{args.max_epoch}",
                config=vars(args),
            )
            wandb.run.summary["uuid"] = str(uuid.uuid4())
            wandb.run.summary["experiment_time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            )

    # def _get_aug_name(self):
    #     augmentations = []
    #     # by default, horizontal flips and translations are always applied
    #     if self.args.random_erase:
    #         augmentations.append("erase")
    #     if self.args.color_jitter:
    #         augmentations.append("jitter")
    #     if self.args.color_aug:
    #         augmentations.append("color")
    #     return "+".join(augmentations) if augmentations else "base"

    def watch_model(self, model):
        """Watch model parameters and gradients"""
        if self.enabled and not self.args.evaluate:
            wandb.watch(model, log="all", log_freq=100)
    
    def log_train_metrics(self, metrics, epoch, batch_idx, trainloader_len):
        """Log training metrics with step based on epoch and batch index"""
        if self.enabled:
            step = epoch * trainloader_len + batch_idx
            wandb.log(metrics, step=step)
    
    def log_test_metrics(self, metrics):
        """Log testing/evaluation metrics"""
        if self.enabled and not self.args.evaluate:  # Only log during training runs
            wandb.log(metrics)
    
    def format_train_metrics(
        self, loss, learning_rate
    ):
        """Format training metrics for logging"""
        return {
            "train/loss": loss,

            "train/learning_rate": learning_rate,
        }
    
    def format_test_metrics(self, loss, epoch_time):
        """Format test metrics for logging"""
        return {
            "test/loss": loss,
            "test/epochTime": epoch_time,

        }

    def log_model_artifact(self, checkpoint_path, name=None):
        """Log a model checkpoint as a W&B artifact"""
        if self.enabled and not self.args.evaluate:
            if name is None:
                try:
                    epoch = checkpoint_path.split("-")[-1]
                    aug_name = self._get_aug_name()
                    name = f"{self.args.arch}_{aug_name}_{self.args.lr:.0e}_{self.args.train_batch_size}_{self.args.optim}_{epoch}"
                except:
                    name = f"{self.args.arch}_checkpoint"

            artifact = wandb.Artifact(name=name, type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f"Logged model artifact '{name}' to W&B")

    def finish(self):
        """End the wandb run"""
        if self.enabled:
            wandb.finish()
