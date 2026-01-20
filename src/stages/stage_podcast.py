#!/usr/bin/env python3
"""
stage_podcast.py
NotebookLM-style podcast generation training stage.

Features:
- Dual AI hosts (HOST_A and HOST_B)
- Optional live user interaction (USER)
- Natural dialogue flow with back-channeling
- Turn-based conversation management
"""

import torch
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets

from .base import BaseStage, StageConfig


@dataclass
class PodcastTurn:
    """Represents a single turn in the podcast."""
    speaker: str  # HOST_A, HOST_B, or USER
    text: str
    emotion: str = "neutral"  # neutral, excited, curious, thoughtful


class PodcastFormatter:
    """Format conversations into podcast-style dialogue."""
    
    SPEAKERS = {
        "HOST_A": "[HOST_A]",  # Main narrator, asks questions
        "HOST_B": "[HOST_B]",  # Responds, adds insights
        "USER": "[USER]",       # Live user input
    }
    
    BACK_CHANNELS = [
        "mmhmm", "right", "exactly", "interesting", 
        "oh wow", "I see", "that makes sense",
    ]
    
    @classmethod
    def format_dialogue(cls, turns: List[PodcastTurn]) -> str:
        """Format turns into training text."""
        lines = []
        for turn in turns:
            speaker_token = cls.SPEAKERS.get(turn.speaker, f"[{turn.speaker}]")
            lines.append(f"{speaker_token} {turn.text}")
        return "\n".join(lines)
    
    @classmethod
    def convert_to_podcast(cls, dialogue: List[str]) -> List[PodcastTurn]:
        """Convert raw dialogue to podcast format with dual hosts."""
        turns = []
        for i, text in enumerate(dialogue):
            # Alternate between hosts
            if i % 2 == 0:
                speaker = "HOST_A"
            else:
                speaker = "HOST_B"
            
            turns.append(PodcastTurn(speaker=speaker, text=text))
            
            # Add occasional back-channeling from other host
            if i > 0 and i % 3 == 0 and len(cls.BACK_CHANNELS) > 0:
                import random
                back_channel = random.choice(cls.BACK_CHANNELS)
                other = "HOST_B" if speaker == "HOST_A" else "HOST_A"
                turns.append(PodcastTurn(
                    speaker=other, 
                    text=back_channel,
                    emotion="supportive"
                ))
        
        return turns
    
    @classmethod
    def insert_user_turn(cls, turns: List[PodcastTurn], 
                         user_text: str, 
                         position: int = -1) -> List[PodcastTurn]:
        """Insert a user question/comment into the podcast."""
        user_turn = PodcastTurn(speaker="USER", text=user_text)
        
        if position == -1:
            # Add at end
            turns.append(user_turn)
            # Host A responds to user
            turns.append(PodcastTurn(
                speaker="HOST_A",
                text="That's a great question! Let me address that.",
            ))
        else:
            turns.insert(position, user_turn)
        
        return turns


class PodcastStage(BaseStage):
    """Podcast generation training - NotebookLM style with dual hosts + user."""
    
    CAPABILITY_NAME = "podcast"
    
    DATASET_PATTERNS = [
        "daily_dialog",
        "mozilla-foundation/common_voice_*",
        "*podcast*",
        "*conversation*",
        "*dialog*",
    ]
    
    def __init__(self, config: StageConfig):
        super().__init__(config)
        self.formatter = PodcastFormatter()
    
    def prepare(self) -> bool:
        """Load model and podcast datasets."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading model from {self.config.base_model_path}")
        
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would load model and podcast datasets")
            return True
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Add special tokens for podcast
            special_tokens = {
                "additional_special_tokens": [
                    "[HOST_A]", "[HOST_B]", "[USER]",
                    "[INTRO]", "[OUTRO]", "[BREAK]",
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Load dialogue datasets
            self.logger.info("Loading podcast/conversational datasets...")
            
            datasets = []
            
            try:
                ds = load_dataset(
                    "daily_dialog",
                    split="train",
                    trust_remote_code=True,
                )
                if self.config.sample_size > 0:
                    ds = ds.select(range(min(self.config.sample_size, len(ds))))
                datasets.append(ds)
                self.logger.info(f"Loaded daily_dialog: {len(ds)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load daily_dialog: {e}")
            
            if datasets:
                self.train_dataset = concatenate_datasets(datasets)
                self.logger.info(f"Total training samples: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare: {e}")
            return False
    
    def _format_sample(self, sample: Dict) -> str:
        """Convert a dataset sample to podcast format."""
        if "dialog" in sample:
            dialogue = sample["dialog"]
            # Convert to podcast turns
            turns = self.formatter.convert_to_podcast(dialogue)
            
            # Occasionally simulate user interaction
            import random
            if random.random() < 0.2:  # 20% of samples
                user_questions = [
                    "Can you explain that more?",
                    "What's an example of that?",
                    "How does that work in practice?",
                    "That's interesting, tell me more!",
                ]
                turns = self.formatter.insert_user_turn(
                    turns, 
                    random.choice(user_questions)
                )
            
            return self.formatter.format_dialogue(turns)
        elif "text" in sample:
            return sample["text"]
        return ""
    
    def train(self) -> Dict[str, Any]:
        """Train on podcast dialogue data with dual hosts."""
        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Simulating podcast training...")
            for epoch in range(self.config.epochs):
                self.logger.info(f"[DRY-RUN] Epoch {epoch+1}/{self.config.epochs}")
                for step in range(10):
                    self.current_step += 1
            return {"success": True, "dry_run": True, "steps": self.current_step}
        
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.warning("No training data, skipping")
            return {"success": True, "steps": 0, "skipped": True}
        
        self.logger.info("Starting podcast training with dual hosts + user...")
        self.logger.info("Speakers: [HOST_A] [HOST_B] [USER]")
        
        from src.training_controller import training_step_hook
        
        total_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for sample in self.train_dataset:
                self.current_step += 1
                
                training_step_hook(
                    self.model, self.optimizer, self.current_step,
                    str(self.checkpoint_dir)
                )
                
                # Format as podcast
                text = self._format_sample(sample)
                if not text:
                    continue
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.current_step % 100 == 0:
                    avg = total_loss / self.current_step
                    self.logger.info(f"Step {self.current_step}, Avg Loss: {avg:.4f}")
                
                if self.current_step % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        return {
            "success": True,
            "steps": self.current_step,
            "final_loss": total_loss / max(self.current_step, 1),
        }


class PodcastInference:
    """Interactive podcast generation with live user input."""
    
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.history: List[PodcastTurn] = []
        self.formatter = PodcastFormatter()
    
    def start_podcast(self, topic: str) -> str:
        """Start a new podcast on a topic."""
        intro = f"[INTRO] Welcome to our podcast! Today we're discussing: {topic}\n"
        intro += "[HOST_A] Let's dive right in. This is a fascinating topic.\n"
        intro += "[HOST_B] Absolutely! I've been looking forward to this discussion."
        
        self.history = [
            PodcastTurn("HOST_A", f"Let's dive right in. {topic} is a fascinating topic."),
            PodcastTurn("HOST_B", "Absolutely! I've been looking forward to this discussion."),
        ]
        
        return intro
    
    def continue_podcast(self) -> str:
        """Generate next podcast turn."""
        context = self.formatter.format_dialogue(self.history)
        
        inputs = self.tokenizer(context, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract new turn
        new_part = response[len(context):]
        
        return new_part
    
    def user_input(self, text: str) -> str:
        """Handle live user input and generate response."""
        # Add user turn
        self.history.append(PodcastTurn("USER", text))
        
        # Generate host response
        context = self.formatter.format_dialogue(self.history)
        context += "\n[HOST_A]"  # Prompt HOST_A to respond
        
        inputs = self.tokenizer(context, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        new_turn = response[len(context):]
        
        # Add to history
        self.history.append(PodcastTurn("HOST_A", new_turn))
        
        return f"[HOST_A] {new_turn}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Podcast Training Stage")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--interactive", action="store_true", 
                        help="Run interactive podcast demo")
    args = parser.parse_args()
    
    if args.interactive:
        # Demo mode
        print("=== Interactive Podcast Demo ===")
        print("This requires a trained podcast model.")
        print("Use: python stage_podcast.py --interactive --base-model /trained/model")
        return 0
    
    config = StageConfig(
        capability_name="podcast",
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    stage = PodcastStage(config)
    return 0 if stage.run().get("success") else 1


if __name__ == "__main__":
    exit(main())
