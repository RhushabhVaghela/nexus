
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

def main():
    parser = argparse.ArgumentParser(description="Voice Pipeline CLI Test Harness")
    parser.add_argument("--clone-sample", type=str, help="Path to a .wav file to clone")
    parser.add_argument("--voice-name", type=str, default="GuestVoice", help="Name for the cloned voice")
    parser.add_argument("--text", type=str, default="Hello everyone, welcome to the show!", help="Text to synthesize")
    parser.add_argument("--vibe", type=str, default="excited", help="Vibe for synthesis")
    parser.add_argument("--out-dir", type=str, default="test_output", help="Output directory")
    
    args = parser.parse_args()
    
    from voice_engine.cloner import voice_cloner
    from voice_engine.registry import voice_registry
    from podcast.synthesizer import synthesize_tts
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Voice Cloning
    if args.clone_sample:
        print(f"--- Cloning Voice: {args.voice_name} from {args.clone_sample} ---")
        dna_path = voice_cloner.clone_voice(args.clone_sample, args.voice_name)
        if dna_path:
            print(f"✅ Voice DNA saved to: {dna_path}")
        else:
            print("❌ Cloning failed.")
            return
    
    # 2. List Available Voices
    print("\n--- Available Voice Personas ---")
    voices = voice_registry.list_voices()
    for name, info in list(voices.items())[:5]: # Show first 5
        print(f" - {name}: {info['description']} ({info['type']})")
    print(" ...")
    
    # 3. Synthesize Test
    voice_to_use = args.voice_name if args.clone_sample else "NATM1"
    print(f"\n--- Synthesizing with Voice: {voice_to_use}, Vibe: {args.vibe} ---")
    
    audio_path = synthesize_tts(
        speaker="Host",
        text=args.text,
        out_dir=out_dir,
        tts_backend="personaplex",
        vibe=args.vibe,
        voice_map={"Host": voice_to_use}
    )
    
    print(f"✅ Synthesis complete: {audio_path}")
    print("\nPipeline verified. You can find the output in the directory specified.")

if __name__ == "__main__":
    main()
