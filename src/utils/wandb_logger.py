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
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project="AML-Coursework",
                name=f"{args.arch}__{args.lossf}_{args.lr:.0e}_{args.train_batch_size}_{args.optim}_{args.max_epoch}",
                config=vars(args),
            )
            wandb.run.summary["uuid"] = str(uuid.uuid4())
            wandb.run.summary["experiment_time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            )

    def watch_model(self, model):
        if self.enabled and not self.args.evaluate:
            wandb.watch(model, log="all", log_freq=100)

    def log_metrics_per_epoch(self,metrics,epoch):
        if self.enabled and not self.args.evaluate:
            wandb.log(metrics,step=epoch)

    def log_train_metrics(self, metrics, epoch, batch_idx, trainloader_len):
        if self.enabled:
            step = epoch * trainloader_len + batch_idx
            wandb.log(metrics, step=step)
    
    def log_test_metrics(self, metrics):
        if self.enabled and not self.args.evaluate:  # Only log during training runs
            wandb.log(metrics)
    
    def format_train_metrics(
        self, loss, learning_rate):
        return {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
        }
    
    def format_test_metrics(self, loss, epoch_time):
        return {
            "test/loss": loss,
            "test/epochTime": epoch_time,
        }

    def log_image(self, image, name):
        if self.enabled and not self.args.evaluate:
            image = wandb.Image(image, caption=name)
            wandb.log({"examples": image})

    def log_model_artifact(self, checkpoint_path, name=None):
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
        if self.enabled:
            wandb.finish()