"""
Training module for ABACR (Adaptive Bidirectional Alignment and Context Regularization).

Implements:
1. Bidirectional negative feedback loss
2. Dynamic context regularization
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class ABACR_Module(pl.LightningModule):
    """PyTorch Lightning implementation of ABACR method."""
    
    def __init__(self, 
                 model_name,
                 learning_rate=5e-5,
                 use_bidirectional_loss=True,
                 use_dynamic_context_reg=True,
                 reference_model_name=None):
        """Initialize ABACR module.
        
        Args:
            model_name: Base model name (e.g., "distilgpt2" or "btlm-3b-8k-base")
            learning_rate: Learning rate for optimizer
            use_bidirectional_loss: Whether to use bidirectional negative feedback
            use_dynamic_context_reg: Whether to use dynamic context regularization
            reference_model_name: Name of reference model for feedback (if None, use copy of base model)
        """
        super(ABACR_Module, self).__init__()
        self.save_hyperparameters()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
        
        if reference_model_name is None:
            reference_model_name = model_name
        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name, trust_remote_code=False)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.learning_rate = learning_rate
        self.use_bidirectional_loss = use_bidirectional_loss
        self.use_dynamic_context_reg = use_dynamic_context_reg
        
        self.bidirect_loss_weight = 0.1
        self.context_reg_weight = 0.01
    
    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def training_step(self, batch, batch_idx):
        """Training step with ABACR loss components."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             labels=input_ids)
        base_loss = outputs.loss
        
        total_loss = base_loss
        loss_components = {"base_loss": base_loss.item()}
        
        if self.use_bidirectional_loss:
            with torch.no_grad():
                ref_outputs = self.reference_model(input_ids=input_ids,
                                                  attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
            
            current_logits = outputs.logits
            
            feedback_loss = self._compute_bidirectional_feedback(
                current_logits, ref_logits, input_ids
            )
            
            total_loss += self.bidirect_loss_weight * feedback_loss
            loss_components["feedback_loss"] = feedback_loss.item()
            
        if self.use_dynamic_context_reg:
            reg_loss = self._compute_context_regularization(input_ids, attention_mask)
            
            total_loss += self.context_reg_weight * reg_loss
            loss_components["reg_loss"] = reg_loss.item()
        
        for name, value in loss_components.items():
            self.log(name, value, on_step=True, prog_bar=True)
        
        self.log("total_loss", total_loss, on_step=True, prog_bar=True)
        return total_loss
    
    def _compute_bidirectional_feedback(self, current_logits, ref_logits, input_ids):
        """Compute bidirectional negative feedback loss.
        
        This compares the model's output distribution with a reference model
        to identify and discourage overconfident or potentially harmful predictions.
        """
        shifted_logits = current_logits[:, :-1, :].contiguous()
        shifted_ref_logits = ref_logits[:, :-1, :].contiguous()
        shifted_labels = input_ids[:, 1:].contiguous()
        
        current_probs = F.softmax(shifted_logits, dim=-1)
        ref_probs = F.softmax(shifted_ref_logits, dim=-1)
        
        ratio = current_probs / (ref_probs + 1e-8)
        
        mask = (ratio > 1.0).float()
        tok_indices = shifted_labels.unsqueeze(-1)
        penalty = mask.gather(-1, tok_indices).squeeze(-1)
        
        feedback_loss = torch.mean((ratio.gather(-1, tok_indices).squeeze(-1) - 1.0) * penalty)
        
        return feedback_loss
    
    def _compute_context_regularization(self, input_ids, attention_mask):
        """Compute dynamic context regularization loss.
        
        This adds controlled noise to position representations to make the model
        more robust to variations in context length and distribution.
        """
        batch_size, seq_len = input_ids.shape
        
        position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        noise_scale = position_ids.float() / seq_len
        
        noise = torch.randn(batch_size, seq_len, device=input_ids.device) * noise_scale * 0.01
        
        noise_mask = attention_mask.float() + noise
        noisy_outputs = self.model(input_ids=input_ids, 
                                   attention_mask=noise_mask.clamp(0, 1))
        
        with torch.no_grad():
            clean_outputs = self.model(input_ids=input_ids, 
                                      attention_mask=attention_mask)
        
        reg_loss = F.mse_loss(noisy_outputs.logits, clean_outputs.logits)
        
        return reg_loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def save_model(self, output_dir):
        """Save the trained model."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

def train_model(model_name, 
               data_loader, 
               output_dir,
               epochs=3, 
               use_bidirectional_loss=True,
               use_dynamic_context_reg=True,
               learning_rate=5e-5,
               reference_model_name=None):
    """Train model using ABACR method.
    
    Args:
        model_name: Name of the base model
        data_loader: DataLoader with training data
        output_dir: Directory to save model and logs
        epochs: Number of training epochs
        use_bidirectional_loss: Whether to use bidirectional feedback
        use_dynamic_context_reg: Whether to use dynamic context regularization
        learning_rate: Learning rate for optimizer
        reference_model_name: Name of reference model (if None, use copy of base model)
        
    Returns:
        trained_model: Trained ABACR model
        train_losses: Dictionary of training losses
    """
    model = ABACR_Module(
        model_name=model_name,
        learning_rate=learning_rate,
        use_bidirectional_loss=use_bidirectional_loss,
        use_dynamic_context_reg=use_dynamic_context_reg,
        reference_model_name=reference_model_name
    )
    
    logger = TensorBoardLogger(save_dir=os.path.join(output_dir, "logs"), name="abacr")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="abacr-{epoch:02d}-{total_loss:.2f}",
        monitor="total_loss",
        save_top_k=1,
        mode="min",
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision for GPU
    )
    
    trainer.fit(model, data_loader)
    
    model.save_model(os.path.join(output_dir, "final_model"))
    
    return model, trainer.callback_metrics

def run_training_experiment(model_name, 
                           texts, 
                           output_dir,
                           batch_size=8,
                           max_length=64,
                           epochs=3,
                           variants=None):
    """Run training experiment with ablation studies.
    
    Args:
        model_name: Name of base model
        texts: List of training texts
        output_dir: Directory to save results
        batch_size: Batch size for training
        max_length: Maximum sequence length
        epochs: Number of training epochs
        variants: List of model variants to train (if None, train all variants)
        
    Returns:
        results: Dictionary of training results for each variant
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    if variants is None:
        variants = {
            "Base": {"use_bidirectional_loss": False, "use_dynamic_context_reg": False},
            "FeedbackOnly": {"use_bidirectional_loss": True, "use_dynamic_context_reg": False},
            "DynContextOnly": {"use_bidirectional_loss": False, "use_dynamic_context_reg": True},
            "FullABACR": {"use_bidirectional_loss": True, "use_dynamic_context_reg": True}
        }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    from preprocess import prepare_dataset
    data_loader, _ = prepare_dataset(texts, model_name, max_length, batch_size)
    
    results = {}
    variant_losses = {}
    
    for variant_name, config in variants.items():
        print(f"\nTraining variant: {variant_name}")
        
        model, metrics = train_model(
            model_name=model_name,
            data_loader=data_loader,
            output_dir=os.path.join(output_dir, variant_name),
            epochs=epochs,
            use_bidirectional_loss=config["use_bidirectional_loss"],
            use_dynamic_context_reg=config["use_dynamic_context_reg"],
        )
        
        results[variant_name] = {
            "model": model,
            "metrics": metrics
        }
        
        final_loss = metrics.get("total_loss")
        variant_losses[variant_name] = final_loss.item() if final_loss is not None else None
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(variant_losses.keys()), y=list(variant_losses.values()))
    plt.ylabel("Training Loss")
    plt.title("Component Ablation Training Loss Comparison")
    plt.tight_layout()
    filename = os.path.join(output_dir, "training_loss_ablation_study.pdf")
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Ablation study plot saved as {filename}")
    plt.close()
    
    return results
