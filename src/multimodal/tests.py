"""
Multimodal Test Prompts
Based on: multimodal_test_prompts.md artifact
"""

def get_test_prompts():
    """Return dictionary of test prompts for different modalities."""
    return {
        "vision": [
            {
                "id": "v1_ui_to_react",
                "input": "assets/test_images/dashboard_ui.png",
                "prompt": "Analyze this UI screenshot. Identify color palette and components. Generate pixel-perfect React component."
            },
            {
                "id": "v2_arch_to_terraform",
                "input": "assets/test_images/aws_architecture.png",
                "prompt": "Identify AWS services from diagram. Write Terraform configuration."
            },
            {
                "id": "v3_algo_whiteboard",
                "input": "assets/test_images/binary_tree_whiteboard.jpg",
                "prompt": "Transcribe whiteboard algorithm. Implement in Python."
            }
        ],
        "video": [
            {
                "id": "vid1_bug_repro",
                "input": "assets/test_videos/glitch_repro.mp4",
                "prompt": "Watch recording. Describe glitch sequence. Hypothesize cause. Suggest React fix."
            },
            {
                "id": "vid2_user_flow",
                "input": "assets/test_videos/checkout_flow.mp4",
                "prompt": "Map user flow. Generate React Navigation stack. List API calls."
            }
        ],
        "audio": [
            {
                "id": "aud1_code_review",
                "input": "assets/test_audio/pr_review_comments.mp3",
                "prompt": "Transcribe feedback on AuthService. Generate diff/apply changes."
            },
            {
                "id": "aud2_sprint_planning",
                "input": "assets/test_audio/sprint_planning.wav",
                "prompt": "Extract user stories. Create JSON tickets. Flag dependencies."
            }
        ]
    }
