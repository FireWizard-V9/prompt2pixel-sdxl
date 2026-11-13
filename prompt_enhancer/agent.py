import os
import uuid
import datetime
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SDXL_SERVICE_URL = os.getenv("SDXL_SERVICE_URL", "http://172.31.23.115:8000")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is missing.")

def generate_timestamped_trace_id() -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}-{short_uuid}"

def truncate_to_clip_limit(prompt_text: str, max_words: int = 60) -> str:
    """Hard programmatic check to prevent CLIP 77-token truncation errors."""
    words = prompt_text.strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return prompt_text

# 1. Initialize LangChain ChatOpenAI pointing to OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://sdxl-agent.internal",
        "X-Title": "LangChain SDXL Prompt Enhancer Agent",
    },
    temperature=0.6
)

# 2. Pydantic Output Schema with Strict Validation
class EnhancedPromptOutput(BaseModel):
    enhanced_prompts: List[str] = Field(
        ..., 
        description="List of concise, domain-accurate SDXL prompts under 60 words."
    )

    @field_validator("enhanced_prompts")
    @classmethod
    def validate_and_truncate_prompts(cls, prompts: List[str]) -> List[str]:
        cleaned = []
        for p in prompts:
            words = p.strip().split()
            if len(words) > 60:
                cleaned.append(" ".join(words[:60]))
            else:
                cleaned.append(p.strip())
        return cleaned

structured_llm = llm.with_structured_output(EnhancedPromptOutput)

# 3. Optimized System Prompt with Concise Few-Shot Examples (35-50 words each)
SYSTEM_PROMPT = """
You are 'VisionCrafter-Agent', an elite Diffusion Prompt Engineer.
Your goal is to convert user concepts into concise, visual SDXL 1.0 prompts.

System Time Context: {current_datetime}

### HARD CONSTRAINTS:
1. **STRICT CLIP LENGTH LIMIT (CRITICAL):**
   Every generated prompt MUST BE BETWEEN 35 AND 55 WORDS (STRICT MAXIMUM 60 WORDS). SDXL truncates anything past 77 tokens. Be dense, direct, and concise.
2. **STRICT USER OVERRIDE RULE:**
   If the user specifies an explicit style (e.g. "anime", "flat line art", "white background"), RESPECT IT STRICTLY. Do not add unwanted realistic tokens.
3. **NO BUZZWORDS:** Avoid "masterpiece", "best quality", "hyperrealistic". Use concrete descriptors.

### CONCISE FEW-SHOT EXAMPLES (35-50 Words):

Example 1 (Chemistry):
User Concept: "Electrolysis of water experiment setup"
Expanded Prompt: "Scientific laboratory photo of a glass Hoffmann voltameter filled with acidified water. Platinum electrodes producing oxygen and hydrogen bubbles at anode and cathode. Clear volumetric marks, lab bench background, soft ring-light illumination, crisp reflections, textbook 3D render quality."

Example 2 (Mathematics):
User Concept: "3D visualization of Fibonacci golden spiral in nature"
Expanded Prompt: "3D glassmorphism artwork of Fibonacci golden spiral geometry overlaying a nautilus shell. Translucent illuminated grid lines in golden ratio proportions. Dark obsidian background, subtle floating particles, cinematic rim lighting in gold and cyan, high-detail mathematical render."

Example 3 (History):
User Concept: "Emperor Ashoka after Kalinga war"
Expanded Prompt: "Dramatic historical oil painting of Emperor Ashoka on the Kalinga battlefield, somber facial expression looking down at his sword. Wearing Mauryan silk sashes and weathered bronze armor. Misty battlefield sunset, distant Ashoka pillar, warm amber side-lighting, expressive brushwork."

Example 4 (Biology):
User Concept: "Structure of human neuron firing signal"
Expanded Prompt: "High-resolution 3D medical illustration of a human neuron. Bright glowing blue action potential travelling down the myelin-sheathed axon. Surrounding astrocytes, dark violet background, bioluminescent neurotransmitters releasing at synaptic cleft, depth of field highlighting cell soma."

Example 5 (Literature):
User Concept: "The lonely cottage from Wordsworth poem in Lake District"
Expanded Prompt: "Landscape painting of a stone cottage with thatched roof in rolling green hills of English Lake District. Wildflowers along cobblestone path, overcast morning sky, gentle mist on mirror-like lake. Romantic watercolor style, soft pastel washes, warm light in windows."
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Generate {count} concise prompt variation(s) (MAX 50 WORDS EACH) for: '{user_concept}'")
])

agent_chain = prompt_template | structured_llm

# 4. FastAPI Setup
app = FastAPI(title="LangChain SDXL Prompt Enhancer Agent Service")

class AgentUserRequest(BaseModel):
    prompt: str = Field(..., description="Raw user query across any academic or creative domain")
    mode: Optional[str] = Field("single", description="Mode: 'single', 'batch', or 'multi-prompt'")
    num_images: Optional[int] = Field(1, description="Number of images/variations")
    steps: Optional[int] = Field(4, description="Inference steps for SDXL (default: 4)")

@app.post("/agent/process")
def process_agent_request(req: AgentUserRequest, x_request_id: Optional[str] = Header(None)):
    trace_id = x_request_id if x_request_id else generate_timestamped_trace_id()
    count = req.num_images if req.mode == "multi-prompt" else 1
    current_dt_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        chain_output: EnhancedPromptOutput = agent_chain.invoke({
            "current_datetime": current_dt_str,
            "user_concept": req.prompt,
            "count": count
        })
        # Double safety truncation
        enhanced_prompts = [truncate_to_clip_limit(p, max_words=60) for p in chain_output.enhanced_prompts]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LangChain Agent Processing Error: {str(e)}")

    if req.mode == "single":
        target_endpoint = f"{SDXL_SERVICE_URL}/generate/single"
        payload = {"prompt": enhanced_prompts[0], "steps": req.steps}
    elif req.mode == "batch":
        target_endpoint = f"{SDXL_SERVICE_URL}/generate/batch"
        payload = {"prompt": enhanced_prompts[0], "num_images": req.num_images, "steps": req.steps}
    elif req.mode == "multi-prompt":
        target_endpoint = f"{SDXL_SERVICE_URL}/generate/multi-prompt"
        payload = {"prompts": enhanced_prompts, "steps": req.steps}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'single', 'batch', or 'multi-prompt'.")

    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": trace_id
    }
    
    try:
        gpu_response = requests.post(target_endpoint, json=payload, headers=headers, timeout=180)
        gpu_response.raise_for_status()
        gpu_result = gpu_response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"GPU Service Inference Failure: {str(e)}")

    return {
        "request_id": trace_id,
        "original_prompt": req.prompt,
        "mode": req.mode,
        "enhanced_prompts_used": enhanced_prompts,
        "gpu_service_response": gpu_result
    }