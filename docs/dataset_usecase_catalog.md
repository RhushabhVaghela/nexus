- - # Your Datasets & Base Models: Use Case & Category Catalog
  
    **Last Updated:** January 20, 2026  
    **Dataset Root:** `E:\data\datasets\`  
    **Base Model Root:** `E:\data\base-model\`
  
    ---
  
    ## 1. Reasoning & Chain-of-Thought (CoT)
  
    ### High Complexity
  
    - **kaist-ai_CoT-Collection**  
      - Focus: Comprehensive chain-of-thought examples  
      - Use: Complex reasoning tasks, multi-step problem solving  
  
    - **O1-OPEN_OpenO1-SFT-Pro**  
      - Focus: Professional-grade O1-style reasoning  
      - Use: Production reasoning pipelines, advanced problem decomposition  
  
    - **O1-OPEN_OpenO1-SFT-Ultra**  
      - Focus: Ultra-advanced reasoning  
      - Use: Cutting-edge reasoning tasks, research-level complexity  
  
    ### Medium Complexity
  
    - **dipta007_APIGen-MT-5k-with-think**  
      - Focus: API generation with intermediate reasoning steps  
      - Use: Tool-calling with reasoning, function synthesis with thought traces  
  
    - **openai_gsm8k**  
      - Focus: Grade school math word problems with solutions  
      - Use: Mathematical reasoning and problem solving  
  
    ### Low Complexity
  
    - **tatsu-lab_alpaca**  
      - Focus: General instruction tuning  
      - Use: Basic instruction following and simple reasoning  
  
    ---
  
    ## 2. Tool Calling & Function Generation
  
    ### High Quality
  
    - **gorilla-llm_Berkeley-Function-Calling-Leaderboard**  
      - Focus: Curated function-calling benchmark  
      - Use: Evaluating function-calling quality, production-style APIs  
  
    - **gorilla-llm_gorilla-openfunctions-v2**  
      - Focus: OpenFunctions v2 with broad API coverage  
      - Use: Multi-domain function calling and tool integration  
  
    ### Medium Quality
  
    - **argilla_apigen-function-calling**  
      - Focus: APIGen function-generation data  
      - Use: Synthetic API definition / function generation  
  
    - **Salesforce_APIGen-MT-5k**  
      - Focus: Multi-turn API generation (≈5k samples)  
      - Use: Multi-step tool chains, conversational tool use  
  
    - **hiyouga_glaive-function-calling-v2-sharegpt**  
      - Focus: ShareGPT-style chats with tools  
      - Use: Naturalistic dialogues with function calls embedded  
  
    - **NousResearch_hermes-function-calling-v1**  
      - Focus: Hermes-style tool calling  
      - Use: General-purpose function-calling SFT  
  
    ### Low Complexity
  
    - **minpeter_xlam-function-calling-60k-parsed**  
      - Focus: Parsed XLAM function-calling logs (≈60k)  
      - Use: Large-scale pattern learning for tool calling  
  
    - **Salesforce_xlam-function-calling-60k**  
      - Focus: XLAM function-calling dataset (≈60k)  
      - Use: Tool-calling behavior at scale  
  
    ---
  
    ## 3. Podcast & Multi-Speaker Dialogue
  
    ### Primary Podcast / Conversation Datasets
  
    - **olewave_OleSpeech-IV-2025-EN-AR-100**  
      - Focus: Multispeaker conversational speech (≈100h EN/AR subset)  
      - Use: Podcast-style dialogue, speaker diarization, host/guest patterns  
      - Format: FLAC audio + refined transcripts + speaker turns  
  
    - **blitt_SPoRC**  
      - Focus: Podcast transcripts with structure  
      - Use: Script-style podcast generation, host/guest roles  
  
    - **spawn99_CornellMovieDialogCorpus**  
      - Focus: Movie dialogues (multi-character)  
      - Use: Personality-rich dialogues, character-driven conversations  
  
    - **IVLLab_MultiDialog**  
      - Focus: Audio-visual dialogues with emotion labels  
      - Use: Expressive, multimodal conversational modeling  
  
    ---
  
    ## 4. Tri-Streaming (Vision + Audio + Text)
  
    ### Multimodal Core
  
    - **E-MM1-100M**  
      - Focus: 100M+ multimodal quintuples  
      - Use: Core tri-/omni-modal pretraining (vision + audio + text + more)  
  
    - **CASIA-IVA-Lab_valor-32k-annotations**  
      - Focus: VALOR-32K audiovisual benchmark  
      - Use: Vision–audio fusion, captioning, evaluation  
  
    - **fullstack__stargate_s04e01_100topkdiverse_text2vid**  
      - Focus: Text-to-video examples (Stargate episode slice)  
      - Use: Video generation and grounding from text context  
  
    - **mvp-lab_LLaVA-OneVision-1.5-RL-Data**  
      - Focus: LLaVA-OneVision RL data  
      - Use: Vision–language alignment and RL-style supervision  
  
    - **OpenGVLab_ShareGPT-4o**  
      - Focus: Multimodal ShareGPT-style logs  
      - Use: Vision + text conversational modeling  
  
    - **VLM2Vec_MSR-VTT**  
      - Focus: Video–text dataset with embeddings  
      - Use: Video–text alignment and retrieval  
  
    - **qingy2024_VaTeX**  
      - Focus: Video + text descriptions  
      - Use: Video captioning and multimodal grounding  
  
    ### Audio-Oriented
  
    - **Mozilla_Common-Voice**  
      - Focus: Large multilingual speech corpus  
      - Use: ASR, speaker diversity, audio encoder training  
  
    - **VoiceAssistant_Lite**  
      - Focus: Voice assistant commands/dialogue  
      - Use: Voice command recognition, simple voice interaction  
  
    ---
  
    ## 5. Dialogue & Relation Extraction
  
    - **nlpdata_dialogre**  
      - Focus: Dialogue-based relation extraction from TV transcripts  
      - Use: Extracting entity relations from multi-turn conversations  
  
    ---
  
    ## 6. Code & Programming
  
    ### High Quality Code Corpora
  
    - **bigcode_commitpackft**  
      - Focus: Commits + code  
      - Use: Code change understanding, commit-message generation  
  
    - **bigcode_the-stack-smol**  
      - Focus: Filtered subset of The Stack  
      - Use: General code pretraining / completion  
  
    - **bigcode_the-stack-smol-xl**  
      - Focus: Larger “smol” variant  
      - Use: Broader-scale code pretraining  
  
    ### Medium / QA-Oriented
  
    - **imoore_60k-stack-overflow-questions-with-quality-rateing**  
      - Focus: StackOverflow Q&A with quality ratings (≈60k)  
      - Use: Programming QA, quality-aware answer generation  
  
    - **pacovaldez_stackoverflow-questions**  
      - Focus: StackOverflow questions (and possibly answers)  
      - Use: Programming knowledge and troubleshooting patterns  
  
    - **samiyasamiya_codegenrate3**  
      - Focus: Code generation benchmark data  
      - Use: Evaluating and training codegen models  
  
    ---
  
    ## 7. Knowledge & Instruction Tuning
  
    ### General Knowledge / Instructions
  
    - **cais_mmlu**  
      - Focus: MMLU benchmark  
      - Use: Multi-domain knowledge evaluation / probing  
  
    - **mrm8488_WebSight_70k**  
      - Focus: Web-sourced multimodal (page + screenshot) data  
      - Use: Web understanding, visual–text grounding  
  
    - **TIGER-Lab_WebInstructSub**  
      - Focus: Web-based instruction subset  
      - Use: General instruction-tuning  
  
    ### Math / Specialized Reasoning
  
    - **AI4Math_IneqMath**  
      - Focus: Inequality-focused math problems  
      - Use: Symbolic and numeric inequality reasoning  
  
    - **AI4Math_MathVerse**  
      - Focus: Diverse math problems & solutions  
      - Use: Broad math reasoning training  
  
    - **AI4Math_MathVista**  
      - Focus: Visual math tasks  
      - Use: Diagram-based and visual math reasoning  
  
    ---
  
    ## 8. Other / Mixed General-Purpose
  
    - **premium_text**  
      - Focus: Curated high-quality text  
      - Use: General language modeling / SFT base  
  
    - **WizardLMTeam_WizardLM_evol_instruct_70k**  
      - Focus: 70k evolution-style instructions  
      - Use: Instruction-following and capability scaling  
  
    ---
  
    ## 9. Base Models (`E:\data\base-model\`)
  
    ### 9.1 Core LLMs & Agents
  
    - **AgentCPM-Explore**  
      - Type: Multimodal / agentic LLM  
      - Role: Experimental backbone for agent-style reasoning and tool use  
      - Use: High-level planning, environment interaction, and reasoning-heavy tasks  
  
    - **PaDT_OVD_3B**  
      - Type: 3B-parameter open-vision-dialogue style model  
      - Role: Lightweight vision-dialogue backbone  
      - Use: Smaller-scale multimodal experiments, vision-text dialogue with limited VRAM footprint  
  
    - **Qwen2.5-Omni-7B-GPTQ-Int4**  
      - Type: 7B omni-modal LLM (GPTQ Int4 quantized)  
      - Role: Main **omni / any-to-any** runtime model on local hardware  
      - Use: Text, vision, and possibly audio-conditioned generation with low VRAM usage  
      - Notes: Good fit for tri-streaming + function calling on RTX 5080 16GB-class GPUs  
  
    ### 9.2 Encoders (Vision & Audio)
  
    - **siglip2-so400m-patch16-512**  
      - Type: Vision encoder (SigLIP2, ~400M)  
      - Role: Image / frame embedding backbone  
      - Use:  
        - Screenshot understanding  
        - Game / video frame encoding  
        - Visual context for tri-streaming  
      - Notes: Pairs with your multimodal JSONL schema for `modalities.image`  
  
    - **whisper-large-v3-turbo**  
      - Type: Audio encoder / ASR model  
      - Role: High-quality transcription and audio features  
      - Use:  
        - Podcast audio to text (from OleSpeech-IV, SPoRC, etc.)  
        - In-game sound / environment audio encoding  
        - User voice command transcription in tri-streaming  
  
    - **parakeet-tdt-0.6b-v3**  
      - Type: TDT-oriented speech / audio model (~0.6B)  
      - Role: Lightweight speech model, possibly for TTS or targeted ASR/diarization tasks  
      - Use:  
        - Faster experiments where full Whisper is overkill  
        - Auxiliary audio modeling or speaker-related tasks  
  
    ---
  
    ## 10. Quick Use-Case Mapping
  
    ### Podcast & Dialogue
  
    - **Datasets**  
      - Primary:  
        - `olewave_OleSpeech-IV-2025-EN-AR-100`  
        - `blitt_SPoRC`  
        - `spawn99_CornellMovieDialogCorpus`  
      - Supporting:  
        - `IVLLab_MultiDialog`  
  
    - **Models**  
      - LLM backbone: `Qwen2.5-Omni-7B-GPTQ-Int4`  
      - Audio encoder: `whisper-large-v3-turbo`  
      - Optional: `parakeet-tdt-0.6b-v3` for lighter speech tasks  
  
    ### Tri-Streaming (Vision + Audio + Text)
  
    - **Datasets**  
      - Core:  
        - `E-MM1-100M`  
      - Supporting / Eval:  
        - `CASIA-IVA-Lab_valor-32k-annotations`  
        - `OpenGVLab_ShareGPT-4o`  
        - `VLM2Vec_MSR-VTT`  
        - `qingy2024_VaTeX`  
      - Audio layer:  
        - `Mozilla_Common-Voice`  
        - `VoiceAssistant_Lite`  
        - `olewave_OleSpeech-IV-2025-EN-AR-100`  
  
    - **Models**  
      - Vision encoder: `siglip2-so400m-patch16-512`  
      - Audio encoder: `whisper-large-v3-turbo`  
      - Core LLM: `Qwen2.5-Omni-7B-GPTQ-Int4`  
      - Agent experiments: `AgentCPM-Explore`  
  
    ### Reasoning (Low / Medium / High)
  
    - High:  
      - `kaist-ai_CoT-Collection`  
      - `O1-OPEN_OpenO1-SFT-Pro`  
      - `O1-OPEN_OpenO1-SFT-Ultra`  
    - Medium:  
      - `dipta007_APIGen-MT-5k-with-think`  
      - `openai_gsm8k`  
    - Low:  
      - `tatsu-lab_alpaca`  
  
    ### Tool Calling
  
    - High:  
      - `gorilla-llm_Berkeley-Function-Calling-Leaderboard`  
      - `gorilla-llm_gorilla-openfunctions-v2`  
    - Medium/Other:  
      - `argilla_apigen-function-calling`  
      - `Salesforce_APIGen-MT-5k`  
      - `hiyouga_glaive-function-calling-v2-sharegpt`  
      - `NousResearch_hermes-function-calling-v1`  
      - `minpeter_xlam-function-calling-60k-parsed`  
      - `Salesforce_xlam-function-calling-60k`  
  
    ### Code
  
    - Core code pretraining:  
      - `bigcode_the-stack-smol`  
      - `bigcode_the-stack-smol-xl`  
      - `bigcode_commitpackft`  
    - QA / Tasks:  
      - `imoore_60k-stack-overflow-questions-with-quality-rateing`  
      - `pacovaldez_stackoverflow-questions`  
      - `samiyasamiya_codegenrate3`  
  
    ---
  
    _Save this as `dataset_and_models_catalog.md` (or update your existing file) in your notes / repo._