# Nexus API Reference

## Overview

The Nexus platform provides a comprehensive REST API for inference, training, and model management operations. This reference documents all available endpoints, request/response schemas, authentication requirements, and usage examples. The API follows OpenAPI 3.0 specifications and is compatible with OpenAI's API format for ease of integration.

The API is organized into logical service groups: Inference API for model serving, Training API for training job management, Model API for model registry operations, and System API for monitoring and administration. All endpoints require authentication via API key or OAuth 2.0 token, with rate limiting applied based on your subscription tier.

This reference is intended for developers building applications on top of Nexus, DevOps teams integrating Nexus into their infrastructure, and system administrators managing API access and quotas. For conceptual guides and tutorials, see the main documentation.

## Installation

### Python Client Installation

```bash
# Install the official Nexus Python client
pip install nexus-client

# For development with extras
pip install nexus-client[dev,torch,transformers]
```

### JavaScript/TypeScript Client Installation

```bash
# npm
npm install @nexus/client

# yarn
yarn add @nexus/client

# pnpm
pnpm add @nexus/client
```

### Go Client Installation

```bash
go get github.com/nexus-ai/client-go
```

### Direct HTTP Access

The API can be accessed directly via HTTP without any client library:

```bash
# Base URL for production
export NEXUS_API_BASE="https://api.nexus.example.com/v1"

# Example cURL request
curl -X POST "${NEXUS_API_BASE}/chat/completions" \
  -H "Authorization: Bearer ${NEXUS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "multimodal-model",
    "messages": [
      {"role": "user", "content": "Analyze this image"}
    ],
    "max_tokens": 512
  }'
```

## Usage

### Python Client Quick Start

```python
from nexus import NexusClient, TextMessage, ImageMessage

# Initialize client with API key
client = NexusClient(
    api_key="your-api-key",
    base_url="https://api.nexus.example.com"
)

# Simple text completion
response = client.completions.create(
    model="text-model",
    prompt="Explain quantum computing in simple terms:",
    max_tokens=500,
    temperature=0.7
)
print(response.choices[0].text)

# Chat completion
chat_response = client.chat.completions.create(
    model="chat-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the key benefits of AI?"}
    ],
    max_tokens=300
)
print(chat_response.choices[0].message.content)

# Multimodal processing
multimodal_response = client.multimodal.generate(
    model="multimodal-model",
    messages=[
        TextMessage(role="user", content="What's in this image?"),
        ImageMessage(role="user", content="https://example.com/image.jpg")
    ]
)
print(multimodal_response.choices[0].message.content)

# Streaming response
for chunk in client.completions.create(
    model="text-model",
    prompt="Write a story about a robot:",
    max_tokens=1000,
    stream=True
):
    print(chunk.choices[0].text, end="", flush=True)
```

### JavaScript/TypeScript Usage

```typescript
import { NexusClient, ChatMessage, ImageContent } from '@nexus/client';

// Initialize client
const client = new NexusClient({
  apiKey: process.env.NEXUS_API_KEY,
  baseUrl: 'https://api.nexus.example.com'
});

// Chat completion
async function chatExample() {
  const response = await client.chat.completions.create({
    model: 'chat-model',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Explain machine learning briefly.' }
    ],
    maxTokens: 512,
    temperature: 0.7
  });
  
  console.log(response.choices[0].message.content);
}

// Streaming example
async function streamExample() {
  const stream = await client.completions.create({
    model: 'text-model',
    prompt: 'List 10 programming languages:',
    maxTokens: 200,
    stream: true
  });
  
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0].text);
  }
}
```

## Inference API

### Completions

#### Create Completion

Generate text completions for a given prompt.

**Endpoint:** `POST /v1/completions`

**Request Parameters:**

```python
from typing import Optional, List, Dict, Any

class CompletionRequest:
    model: str                              # Model ID to use
    prompt: str                             # Input prompt
    suffix: Optional[str] = None            # Text after the completion
    max_tokens: Optional[int] = None        # Maximum tokens to generate
    temperature: Optional[float] = 1.0      # Sampling temperature (0-2)
    top_p: Optional[float] = 1.0            # Top-p sampling (0-1)
    n: Optional[int] = 1                    # Number of completions
    stream: Optional[bool] = False          # Enable streaming
    logprobs: Optional[bool] = False        # Return log probabilities
    echo: Optional[bool] = False            # Echo prompt in output
    stop: Optional[List[str]] = None        # Stop sequences
    presence_penalty: Optional[float] = 0   # Presence penalty (-2 to 2)
    frequency_penalty: Optional[float] = 0  # Frequency penalty (-2 to 2)
    best_of: Optional[int] = 1              # Generate best_of completions server-side
    logit_bias: Optional[Dict[str, float]] = None  # Token bias
    user: Optional[str] = None              # User identifier for tracking
```

**Response:**

```python
class CompletionResponse:
    id: str                                 # Unique completion ID
    object: str                             # Object type ("text_completion")
    created: int                            # Unix timestamp
    model: str                              # Model ID used
    choices: List[CompletionChoice]         # Generated completions
    usage: UsageInfo                        # Token usage statistics

class CompletionChoice:
    text: str                               # Generated text
    index: int                              # Completion index
    logprobs: Optional[LogProbInfo]         # Log probabilities if requested
    finish_reason: str                      # Reason for stopping
    stop_reason: Optional[int] = None       # Token index where stopped

class UsageInfo:
    prompt_tokens: int                      # Tokens in prompt
    completion_tokens: int                  # Tokens in completion
    total_tokens: int                       # Total tokens used
```

**Example Request:**

```bash
curl -X POST "https://api.nexus.example.com/v1/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-model-llama-3-8b",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "n": 3,
    "stop": ["\n\n", "."]
  }'
```

**Example Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1699999999,
  "model": "text-model-llama-3-8b",
  "choices": [
    {
      "text": " incredibly promising, with applications spanning healthcare, education, and scientific research.",
      "index": 0,
      "finish_reason": "stop",
      "stop_reason": 15
    },
    {
      "text": " transforming how we work and live, automating complex tasks and enabling new forms of creativity.",
      "index": 1,
      "finish_reason": "stop",
      "stop_reason": 18
    },
    {
      "text": " rapidly evolving, with breakthroughs in reasoning, planning, and multimodal understanding.",
      "index": 2,
      "finish_reason": "stop",
      "stop_reason": 16
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 49,
    "total_tokens": 57
  }
}
```

### Chat Completions

#### Create Chat Completion

Generate chat-based responses for conversational AI applications.

**Endpoint:** `POST /v1/chat/completions`

**Request Parameters:**

```python
from typing import Optional, List, Dict, Any, Union

class ChatCompletionRequest:
    model: str                              # Model ID to use
    messages: List[ChatMessage]             # Conversation messages
    frequency_penalty: Optional[float] = 0  # Frequency penalty (-2 to 2)
    logit_bias: Optional[Dict[str, float]] = None  # Token bias
    logprobs: Optional[bool] = False        # Return log probabilities
    top_logprobs: Optional[int] = None      # Number of top logprobs
    max_tokens: Optional[int] = None        # Max tokens to generate
    n: Optional[int] = 1                    # Number of completions
    presence_penalty: Optional[float] = 0   # Presence penalty (-2 to 2)
    response_format: Optional[Dict] = None  # Response format (JSON mode)
    seed: Optional[int] = None              # Random seed for reproducibility
    stop: Optional[Union[str, List[str]]] = None  # Stop sequences
    stream: Optional[bool] = False          # Enable streaming
    temperature: Optional[float] = 1.0      # Temperature (0-2)
    top_p: Optional[float] = 1.0            # Top-p sampling (0-1)
    tools: Optional[List[Tool]] = None      # Available tools
    tool_choice: Optional[Union[str, ToolChoice]] = None  # Tool selection
    user: Optional[str] = None              # User identifier

class ChatMessage:
    role: str                               # "system", "user", "assistant", "tool"
    content: Union[str, List[ContentPart]]  # Message content
    name: Optional[str] = None              # Message author name
    tool_calls: Optional[List[ToolCall]] = None  # Tool calls (assistant)

class ContentPart:
    type: str                               # "text", "image_url", "input_audio"
    text: Optional[str] = None              # Text content
    image_url: Optional[ImageURL] = None    # Image content
    input_audio: Optional[InputAudio] = None  # Audio content

class ImageURL:
    url: str                                # Image URL or base64 data
    detail: Optional[str] = "auto"          # "low", "high", "auto"
```

**Response:**

```python
class ChatCompletionResponse:
    id: str                                 # Unique completion ID
    object: str                             # Object type ("chat.completion")
    created: int                            # Unix timestamp
    model: str                              # Model ID used
    system_fingerprint: str                 # Server configuration fingerprint
    choices: List[ChatCompletionChoice]     # Generated responses
    usage: UsageInfo                        # Token usage statistics

class ChatCompletionChoice:
    index: int                              # Choice index
    message: ChatMessage                    # Generated message
    finish_reason: str                      # Stop reason
    logprobs: Optional[LogProbInfo] = None  # Log probabilities
    stop_reason: Optional[int] = None       # Token index where stopped
```

**Example Request:**

```bash
curl -X POST "https://api.nexus.example.com/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat-model-llama-3-70b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful coding assistant."
      },
      {
        "role": "user",
        "content": "Write a Python function to calculate factorial"
      }
    ],
    "max_tokens": 200,
    "temperature": 0.2,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "execute_code",
          "description": "Execute Python code in a sandbox",
          "parameters": {
            "type": "object",
            "properties": {
              "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
          }
        }
      }
    ]
  }'
```

**Example Response:**

```json
{
  "id": "chatcmpl-xyz789",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "chat-model-llama-3-70b",
  "system_fingerprint": "fp_12345",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'll create a factorial function for you. Would you like me to also execute it to verify?",
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "execute_code",
              "arguments": "{\"code\": \"def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n - 1)\\n\\nprint(f\\\"5! = {factorial(5)}\\\")\\nprint(f\\\"10! = {factorial(10)}\\\")\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 35,
    "completion_tokens": 89,
    "total_tokens": 124
  }
}
```

### Embeddings

#### Create Embeddings

Generate vector embeddings for text input.

**Endpoint:** `POST /v1/embeddings`

**Request Parameters:**

```python
class EmbeddingRequest:
    model: str                              # Embedding model ID
    input: Union[str, List[str]]            # Input text(s) to embed
    encoding_format: Optional[str] = "float"  # "float" or "base64"
    dimensions: Optional[int] = None         # Reduce dimensions
    user: Optional[str] = None              # User identifier
```

**Response:**

```python
class EmbeddingResponse:
    object: str                             # Object type ("list")
    data: List[EmbeddingData]               # Embedding vectors
    model: str                              # Model ID used
    usage: UsageInfo                        # Token usage

class EmbeddingData:
    object: str                             # Object type ("embedding")
    embedding: List[float]                  # Embedding vector
    index: int                              # Input index
```

**Example Request:**

```bash
curl -X POST "https://api.nexus.example.com/v1/embeddings" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embedding-model-e5-large",
    "input": [
      "The cat sat on the mat",
      "The dog ran in the park"
    ],
    "encoding_format": "float",
    "dimensions": 512
  }'
```

### Multimodal

#### Generate Multimodal Output

Process and generate responses for multimodal inputs (text, images, audio).

**Endpoint:** `POST /v1/multimodal/generate`

**Request Parameters:**

```python
class MultimodalGenerationRequest:
    model: str                              # Multimodal model ID
    messages: List[MultimodalMessage]       # Input messages with content parts
    max_tokens: Optional[int] = None        # Max tokens to generate
    temperature: Optional[float] = 1.0      # Temperature (0-2)
    top_p: Optional[float] = 1.0            # Top-p sampling
    stop: Optional[List[str]] = None        # Stop sequences
    stream: Optional[bool] = False          # Enable streaming

class MultimodalMessage:
    role: str                               # Message role
    content: List[ContentPart]              # Content parts (text, image, audio)
```

**Example Request with Image:**

```bash
curl -X POST "https://api.nexus.example.com/v1/multimodal/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "multimodal-model-llava",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe what you see in this image:"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/sample-image.jpg",
              "detail": "high"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
```

## Training API

### Training Jobs

#### Create Training Job

Initiate a new training job with specified configuration.

**Endpoint:** `POST /v1/training/jobs`

**Request Parameters:**

```python
from typing import Optional, Dict, Any
from enum import Enum

class TrainingType(str, Enum):
    DPO = "dpo"                             # Direct Preference Optimization
    ORPO = "orpo"                           # Odds Ratio Preference Optimization
    SFT = "sft"                             # Supervised Fine-Tuning
    GRPO = "grpo"                           # Group Relative Preference Optimization

class TrainingJobRequest:
    name: str                               # Job name
    training_type: TrainingType             # Type of training
    model: str                              # Base model ID
    dataset: str                            # Dataset ID or path
    config: TrainingConfig                  # Training configuration
    
class TrainingConfig:
    learning_rate: float                    # Learning rate (e.g., 5e-7)
    batch_size: int                         # Batch size per device
    epochs: int                             # Number of training epochs
    max_seq_length: int = 2048              # Maximum sequence length
    warmup_ratio: float = 0.1               # Warmup ratio
    weight_decay: float = 0.01              # Weight decay
    gradient_accumulation_steps: int = 1    # Gradient accumulation
    optimizer: str = "adamw"                # Optimizer type
    scheduler: str = "cosine"               # LR scheduler
    mixed_precision: bool = True            # Use BF16/FP16
    deepspeed_config: Optional[Dict] = None # DeepSpeed configuration
    lora_config: Optional[LoRAConfig] = None  # LoRA configuration
    fsdp_config: Optional[FSDPConfig] = None  # FSDP configuration

class LoRAConfig:
    r: int = 16                             # LoRA rank
    alpha: int = 32                         # LoRA alpha
    dropout: float = 0.05                   # LoRA dropout
    target_modules: List[str]               # Target modules

class FSDPConfig:
    mixed_precision: bool = True            # Mixed precision
    backward_prefetch: str = "pre_forward"  # Backward prefetch
    forward_prefetch: bool = True           # Forward prefetch
```

**Response:**

```python
class TrainingJobResponse:
    id: str                                 # Job ID
    name: str                               # Job name
    status: str                             # Job status
    created_at: int                         # Creation timestamp
    started_at: Optional[int] = None        # Start timestamp
    finished_at: Optional[int] = None       # Finish timestamp
    config: TrainingConfig                  # Job configuration
    progress: TrainingProgress              # Current progress
    logs: TrainingLogs                      # Training logs URL

class TrainingProgress:
    epoch: int                              # Current epoch
    step: int                               # Current step
    total_steps: int                        # Total steps
    loss: float                             # Current loss
    eval_loss: Optional[float] = None       # Evaluation loss
    learning_rate: float                    # Current learning rate
    throughput: float                       # Samples/second
```

**Example Request:**

```bash
curl -X POST "https://api.nexus.example.com/v1/training/jobs" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "dpo-fine-tune-llama-3-8b",
    "training_type": "dpo",
    "model": "meta-llama/Llama-3-8b-instruct",
    "dataset": "nexus-preference-dataset-v1",
    "config": {
      "learning_rate": 5e-7,
      "batch_size": 4,
      "epochs": 3,
      "max_seq_length": 4096,
      "mixed_precision": true,
      "lora_config": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
      }
    }
  }'
```

#### List Training Jobs

Retrieve list of training jobs.

**Endpoint:** `GET /v1/training/jobs`

**Query Parameters:**

```python
class ListJobsParams:
    status: Optional[str] = None            # Filter by status
    model: Optional[str] = None             # Filter by model
    created_after: Optional[int] = None     # Filter by creation time
    created_before: Optional[int] = None    # Filter by creation time
    limit: int = 20                         # Results per page
    offset: int = 0                         # Pagination offset
```

#### Get Training Job Details

Retrieve detailed information about a training job.

**Endpoint:** `GET /v1/training/jobs/{job_id}`

#### Cancel Training Job

Stop a running training job.

**Endpoint:** `DELETE /v1/training/jobs/{job_id}`

#### Get Training Job Logs

Retrieve training logs.

**Endpoint:** `GET /v1/training/jobs/{job_id}/logs`

**Query Parameters:**

```python
class GetLogsParams:
    tail: Optional[int] = None              # Number of lines from end
    follow: bool = False                    # Stream logs in real-time
    since: Optional[int] = None             # Lines after timestamp
```

### Datasets

#### List Datasets

Retrieve available datasets.

**Endpoint:** `GET /v1/datasets`

#### Upload Dataset

Upload a custom dataset.

**Endpoint:** `POST /v1/datasets`

**Request:** `multipart/form-data`

```python
# Form fields
file: File                                 # Dataset file (JSON, CSV, Parquet)
name: str                                  # Dataset name
format: str                                # Format ("json", "csv", "parquet")
schema: Optional[DatasetSchema] = None     # Dataset schema definition

class DatasetSchema:
    columns: List[ColumnSchema]            # Column definitions
    text_column: str                       # Text column name
    preference_columns: Optional[PreferenceColumns] = None

class PreferenceColumns:
    chosen: str                            # Chosen response column
    rejected: str                          # Rejected response column
```

#### Get Dataset Details

Retrieve dataset information.

**Endpoint:** `GET /v1/datasets/{dataset_id}`

#### Delete Dataset

Remove a dataset.

**Endpoint:** `DELETE /v1/datasets/{dataset_id}`

## Model API

### List Models

Retrieve available models.

**Endpoint:** `GET /v1/models`

**Response:**

```python
class ModelsResponse:
    object: str                             # Object type ("list")
    data: List[ModelInfo]                   # Model list

class ModelInfo:
    id: str                                 # Model ID
    object: str                             # Object type ("model")
    created: int                            # Creation timestamp
    owned_by: str                           # Organization/owner
    permission: List[ModelPermission]       # Access permissions
    root_model: Optional[str] = None        # Root model ID
    parent_model: Optional[str] = None      # Parent model ID

class ModelPermission:
    id: str                                 # Permission ID
    object: str                             # Object type ("model_permission")
    created: int                            # Creation timestamp
    allow_create_engine: bool               # Allow engine creation
    allow_sampling: bool                    # Allow sampling
    allow_logprobs: bool                    # Allow logprobs
    allow_search_indices: bool              # Allow search
    allow_view: bool                        # Allow viewing
    allow_fine_tuning: bool                 # Allow fine-tuning
    allow_system_prompts: bool              # Allow system prompts
    organization: str                       # Organization ID
    group: Optional[str] = None             # Group ID
```

### Get Model Details

Retrieve detailed model information.

**Endpoint:** `GET /v1/models/{model_id}`

**Response:**

```python
class ModelDetails:
    id: str                                 # Model ID
    object: str                             # Object type ("model")
    created: int                            # Creation timestamp
    owned_by: str                           # Owner
    capabilities: ModelCapabilities         # Model capabilities
    pricing: ModelPricing                   # Usage pricing
    context_length: int                     # Max context length
    max_output_tokens: int                  # Max output tokens
    parameters: ModelParameters             # Model parameters

class ModelCapabilities:
    completion_chat: bool                   # Chat completion support
    completion: bool                        # Text completion support
    embeddings: bool                        # Embedding support
    multimodal: bool                        # Multimodal support
    audio: bool                             # Audio support
    video: bool                             # Video support
    fine_tuning: bool                       # Fine-tuning support

class ModelPricing:
    prompt_tokens: str                      # Price per 1M prompt tokens
    completion_tokens: str                  # Price per 1M completion tokens
    image_tokens: str                       # Price per 1M image tokens
```

### Fine-tune Model

Create a fine-tuned model.

**Endpoint:** `POST /v1/models/{model_id}/fine-tunes`

**Request Parameters:**

```python
class FineTuneRequest:
    training_file: str                      # Training file ID
    validation_file: Optional[str] = None   # Validation file ID
    model: str                              # Base model ID
    n_epochs: int = 4                       # Number of epochs
    batch_size: Optional[int] = None        # Batch size
    learning_rate_multiplier: float = 1.0   # LR multiplier
    lora_r: Optional[int] = None            # LoRA rank
    lora_alpha: Optional[int] = None        # LoRA alpha
    lora_dropout: Optional[float] = None    # LoRA dropout
```

## System API

### Health Check

Check API health status.

**Endpoint:** `GET /v1/health`

**Response:**

```python
class HealthResponse:
    status: str                             # "healthy", "degraded", "unhealthy"
    version: str                            # API version
    timestamp: int                          # Unix timestamp
    components: Dict[str, ComponentHealth]  # Component statuses

class ComponentHealth:
    status: str                             # Component status
    latency_ms: float                       # Component latency
    details: Optional[Dict] = None          # Additional details
```

### API Version

Get API version information.

**Endpoint:** `GET /v1/version`

### Rate Limit Status

Check current rate limit status.

**Endpoint:** `GET /v1/rate-limit-status`

**Response:**

```python
class RateLimitResponse:
    object: str                             # Object type
    rate_limits: List[RateLimit]            # Rate limit info

class RateLimit:
    category: str                           # Rate limit category
    limit: int                              # Request limit
    remaining: int                          # Remaining requests
    reset_seconds: int                      # Seconds until reset
```

## WebSocket API

### Streaming Completions

Connect to WebSocket for real-time streaming.

**Endpoint:** `WS /v1/ws/completions`

**Connection URL:**

```
wss://api.nexus.example.com/v1/ws/completions?token=YOUR_API_KEY
```

**Message Formats:**

```python
# Client to Server - Start Request
{
    "type": "start",
    "request_id": "req_123",
    "model": "text-model",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
}

# Server to Client - Token Update
{
    "type": "token",
    "request_id": "req_123",
    "token": " once",
    "logprob": -0.523
}

# Server to Client - Completion
{
    "type": "complete",
    "request_id": "req_123",
    "finish_reason": "stop",
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 45,
        "total_tokens": 50
    }
}

# Server to Client - Error
{
    "type": "error",
    "request_id": "req_123",
    "error": {
        "code": "rate_limit_exceeded",
        "message": "Rate limit exceeded"
    }
}
```

## Error Handling

### Error Response Format

```python
class ErrorResponse:
    object: str                             # Object type ("error")
    message: str                            # Error message
    type: str                               # Error type
    code: str                               # Error code
    param: Optional[str] = None             # Related parameter
    details: Optional[Dict] = None          # Additional details
```

### Common Error Codes

```python
ERROR_CODES = {
    # Authentication Errors
    "unauthenticated": (401, "Missing or invalid API key"),
    "invalid_api_key": (401, "Invalid API key provided"),
    "token_expired": (401, "API token has expired"),
    
    # Authorization Errors
    "permission_denied": (403, "Insufficient permissions"),
    "model_access_denied": (403, "Model access not allowed"),
    "quota_exceeded": (403, "Usage quota exceeded"),
    
    # Request Errors
    "invalid_request_error": (400, "Invalid request parameters"),
    "invalid_encoding": (400, "Invalid encoding format"),
    "context_length_exceeded": (400, "Context too long"),
    "rate_limit_exceeded": (429, "Rate limit exceeded"),
    
    # Server Errors
    "internal_server_error": (500, "Internal server error"),
    "model_not_available": (503, "Model temporarily unavailable"),
    "service_unavailable": (503, "Service temporarily unavailable"),
    
    # Not Found Errors
    "not_found": (404, "Resource not found"),
    "model_not_found": (404, "Model not found"),
    "dataset_not_found": (404, "Dataset not found"),
    "job_not_found": (404, "Training job not found"),
}
```

### Example Error Response

```json
{
  "object": "error",
  "message": "Rate limit exceeded. Please retry in 30 seconds.",
  "type": "rate_limit_error",
  "code": "rate_limit_exceeded",
  "param": null,
  "details": {
    "limit": 1000,
    "remaining": 0,
    "reset_at": 1699999999
  }
}
```

## Rate Limiting

### Rate Limit Tiers

```python
class RateLimitTier:
    FREE = {
        "requests_per_minute": 60,
        "tokens_per_minute": 10000,
        "concurrent_requests": 3
    }
    
    DEVELOPER = {
        "requests_per_minute": 600,
        "tokens_per_minute": 100000,
        "concurrent_requests": 10
    }
    
    TEAM = {
        "requests_per_minute": 3000,
        "tokens_per_minute": 500000,
        "concurrent_requests": 50
    }
    
    ENTERPRISE = {
        "requests_per_minute": 10000,
        "tokens_per_minute": 2000000,
        "concurrent_requests": 200
    }
```

## Authentication

### API Key Authentication

```python
# Header authentication
curl -H "Authorization: Bearer YOUR_API_KEY" ...

# Python client
client = NexusClient(api_key="YOUR_API_KEY")

# JavaScript client
const client = new NexusClient({ apiKey: 'YOUR_API_KEY' })
```

### OAuth 2.0 Authentication

```python
# Get access token
from nexus.auth import NexusOAuth

oauth = NexusOAuth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    redirect_uri="https://your-app.com/callback"
)

# Get authorization URL
auth_url = oauth.get_authorization_url(scope="read write")

# Exchange code for token
token = oauth.exchange_code_for_token(code)
client = NexusClient(token=token)
```

### Token Refresh

```python
# Automatic token refresh
client = NexusClient(
    api_key="YOUR_API_KEY",
    auto_refresh=True,
    refresh_callback=save_new_token
)

# Manual refresh
new_token = client.refresh_token()
```

## Python API Reference

### NexusClient

```python
from nexus import NexusClient

class NexusClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        base_url: str = "https://api.nexus.example.com",
        timeout: float = 60.0,
        max_retries: int = 3,
        auto_refresh: bool = False,
        refresh_callback: Optional[Callable] = None
    ):
        """Initialize Nexus API client.
        
        Args:
            api_key: API key for authentication
            token: OAuth token for authentication
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            auto_refresh: Auto-refresh OAuth tokens
            refresh_callback: Callback for token refresh
        """

    @property
    def completions(self) -> CompletionClient:
        """Text completion client."""
        
    @property
    def chat(self) -> ChatClient:
        """Chat completion client."""
        
    @property
    def embeddings(self) -> EmbeddingClient:
        """Embedding client."""
        
    @property
    def multimodal(self) -> MultimodalClient:
        """Multimodal processing client."""
        
    @property
    def training(self) -> TrainingClient:
        """Training job management client."""
        
    @property
    def models(self) -> ModelClient:
        """Model management client."""
        
    @property
    def datasets(self) -> DatasetClient:
        """Dataset management client."""
        
    async def close(self) -> None:
        """Close the client session."""
```

### CompletionClient

```python
class CompletionClient:
    def __init__(self, client: NexusClient):
        """Completion client.
        
        Args:
            client: NexusClient instance
        """
        
    def create(
        self,
        model: str,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stream: bool = False,
        logprobs: bool = False,
        echo: bool = False,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        best_of: int = 1,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None
    ) -> CompletionResponse:
        """Create text completion.
        
        Args:
            model: Model ID to use
            prompt: Input prompt
            suffix: Text after completion
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            n: Number of completions
            stream: Enable streaming
            logprobs: Return log probabilities
            echo: Echo prompt in output
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            best_of: Generate best_of completions
            logit_bias: Token bias mapping
            user: User identifier
            
        Returns:
            CompletionResponse with generated text
        """
        
    def create_streaming(
        self,
        model: str,
        prompt: str,
        **kwargs
    ) -> Iterator[CompletionChunk]:
        """Create streaming text completion.
        
        Returns:
            Iterator of completion chunks
        """
```

### ChatClient

```python
class ChatClient:
    def __init__(self, client: NexusClient):
        """Chat completion client."""
        
    def create(
        self,
        model: str,
        messages: List[ChatMessage],
        frequency_penalty: float = 0,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        presence_penalty: float = 0,
        response_format: Optional[Dict] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        user: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Create chat completion.
        
        Args:
            model: Model ID to use
            messages: Conversation messages
            frequency_penalty: Frequency penalty
            logit_bias: Token bias
            logprobs: Return log probabilities
            top_logprobs: Number of top logprobs
            max_tokens: Maximum tokens to generate
            n: Number of completions
            presence_penalty: Presence penalty
            response_format: Response format (JSON mode)
            seed: Random seed
            stop: Stop sequences
            stream: Enable streaming
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Available tools
            tool_choice: Tool selection mode
            user: User identifier
            
        Returns:
            ChatCompletionResponse with generated message
        """
```

### TrainingClient

```python
class TrainingClient:
    def __init__(self, client: NexusClient):
        """Training job management client."""
        
    def create_job(
        self,
        name: str,
        training_type: TrainingType,
        model: str,
        dataset: str,
        config: TrainingConfig
    ) -> TrainingJobResponse:
        """Create a new training job.
        
        Args:
            name: Job name
            training_type: Type of training
            model: Base model ID
            dataset: Dataset ID or path
            config: Training configuration
            
        Returns:
            TrainingJobResponse with job details
        """
        
    def list_jobs(
        self,
        status: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[TrainingJobResponse]:
        """List training jobs.
        
        Args:
            status: Filter by status
            model: Filter by model
            limit: Results per page
            offset: Pagination offset
            
        Returns:
            List of TrainingJobResponse
        """
        
    def get_job(self, job_id: str) -> TrainingJobResponse:
        """Get training job details.
        
        Args:
            job_id: Job ID
            
        Returns:
            TrainingJobResponse with job details
        """
        
    def cancel_job(self, job_id: str) -> TrainingJobResponse:
        """Cancel a running training job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated TrainingJobResponse
        """
        
    def get_logs(
        self,
        job_id: str,
        tail: Optional[int] = None,
        follow: bool = False
    ) -> Union[str, Iterator[str]]:
        """Get training job logs.
        
        Args:
            job_id: Job ID
            tail: Number of lines from end
            follow: Stream logs in real-time
            
        Returns:
            Log string or iterator of log lines
        """
```

## Examples

### Basic Text Completion

```python
from nexus import NexusClient

client = NexusClient(api_key="your-api-key")

response = client.completions.create(
    model="text-model-llama-3-8b",
    prompt="Explain the concept of machine learning:",
    max_tokens=300,
    temperature=0.7
)

print(response.choices[0].text)
```

### Conversation with History

```python
from nexus import NexusClient, ChatMessage

client = NexusClient(api_key="your-api-key")

messages = [
    ChatMessage(role="system", content="You are a helpful coding assistant."),
    ChatMessage(role="user", content="What is recursion?"),
    ChatMessage(role="assistant", 
                content="Recursion is when a function calls itself..."),
    ChatMessage(role="user", content="Give me a Python example")
]

response = client.chat.completions.create(
    model="chat-model-llama-3-70b",
    messages=messages,
    max_tokens=200,
    temperature=0.5
)

print(response.choices[0].message.content)
```

### Multimodal Analysis

```python
from nexus import NexusClient, TextMessage, ImageMessage

client = NexusClient(api_key="your-api-key")

response = client.multimodal.generate(
    model="multimodal-model-llava",
    messages=[
        TextMessage(
            role="user", 
            content="What objects can you identify in this image?"
        ),
        ImageMessage(
            role="user",
            content="https://example.com/diagram.png"
        )
    ],
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Fine-tuning Setup

```python
from nexus import NexusClient, TrainingType

client = NexusClient(api_key="your-api-key")

# Create training job
job = client.training.create_job(
    name="dpo-fine-tune-llama3",
    training_type=TrainingType.DPO,
    model="meta-llama/Llama-3-8b-instruct",
    dataset="preference-dataset-v1",
    config=TrainingConfig(
        learning_rate=5e-7,
        batch_size=4,
        epochs=3,
        lora_config=LoRAConfig(r=32, alpha=64)
    )
)

print(f"Job created: {job.id}")

# Monitor progress
while job.status not in ["completed", "failed", "cancelled"]:
    job = client.training.get_job(job.id)
    print(f"Progress: {job.progress.epoch}/{job.config.epochs} epochs, "
          f"Loss: {job.progress.loss:.4f}")
    time.sleep(30)
```

### Streaming Response

```python
from nexus import NexusClient

client = NexusClient(api_key="your-api-key")

print("Generating story...")
for chunk in client.completions.create(
    model="text-model-llama-3-8b",
    prompt="Write a short story about a robot:",
    max_tokens=500,
    temperature=0.8,
    stream=True
):
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
print()
```

### Batch Processing

```python
from nexus import NexusClient, BatchRequest, BatchTask

client = NexusClient(api_key="your-api-key")

# Create batch request
batch = client.batches.create(
    tasks=[
        BatchTask(
            custom_id="item-1",
            model="text-model",
            body={
                "prompt": "Translate to French: Hello",
                "max_tokens": 50
            }
        ),
        BatchTask(
            custom_id="item-2",
            model="text-model",
            body={
                "prompt": "Translate to Spanish: Hello",
                "max_tokens": 50
            }
        )
    ],
    completion_window="24h"
)

# Check status
result = client.batches.get(batch.id)
print(f"Batch status: {result.status}")
```

## See Also

- **[Architecture Overview](ARCHITECTURE.md)** - System architecture details
- **[Pipeline Guide](PIPELINE_GUIDE.md)** - Pipeline configuration
- **[Security Documentation](SECURITY.md)** - Authentication and authorization
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[Training Methods](TRAINING_METHODS.md)** - Training pipeline documentation
- **[API Documentation](https://api.nexus.example.com/docs)** - Interactive API docs
