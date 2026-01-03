# ACE-Step Playground - Development Specification

## Overview

The Playground is a Gradio-based UI for testing and interacting with the ACE-Step music generation pipeline. It consists of two main sections:

1. **LLM Section** - For generating audio codes from text descriptions using the 5Hz LM model
2. **ACEStep Section** - For generating audio from codes using the DiT model

**Key Principles:**
- Keep logic simple - no automatic data flow between sections
- Dynamic UI based on task selection
- Advanced parameters in collapsible accordions
- Only modify playground module files, do not touch existing acestep modules

---

## File Structure

```
playground/
├── playground.py           # Entry point
├── playground_handler.py   # Business logic wrapper
├── playground_ui.py        # Gradio UI definition
└── playground.md           # This specification
```

---

## 1. LLM Section

The LLM Section generates audio codes from text descriptions using the 5Hz Language Model.

### 1.1 Model Loading Sub-Section

| Component | Type | Description |
|-----------|------|-------------|
| `llm_model_dropdown` | Dropdown | Available LLM models from `handler.get_available_llm_models()`, allow custom value, scale=3 |
| `llm_backend` | Dropdown | Options: `["vllm", "pt"]`, default: `"vllm"`, scale=1 |
| `llm_device` | Dropdown | Options: `["auto", "cuda", "cpu"]`, default: `"auto"`, scale=1 |
| `load_llm_btn` | Button (primary) | Triggers `handler.initialize_llm()`, scale=1 |
| `llm_status` | Textbox (read-only) | Displays loading status, placeholder: "LLM not loaded" |

**Handler Method:**
```python
handler.initialize_llm(lm_model_path: str, backend: str, device: str) -> str
```

**Implementation Details:**
- Auto-detects checkpoint directory from project root
- Calls `LLMHandler.initialize()` with proper parameters
- Returns status message string

---

### 1.2 Input Sub-Section

#### 1.2.1 Layout Structure

Two-column layout with `gr.Row`:
- **Left Column (scale=1)**: Text inputs (caption, lyrics, negative_caption, negative_lyrics)
- **Right Column (scale=1)**: Meta group + Config accordion

#### 1.2.2 Text Inputs (Left Column)

| Component | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | Textbox | `""` | Music description, 3 lines |
| `lyrics` | Textbox | `""` | Song lyrics, 5 lines |
| `negative_caption` | Textbox | `"NO USER INPUT"` | Negative prompt for caption, 3 lines |
| `negative_lyrics` | Textbox | `"NO USER INPUT"` | Negative prompt for lyrics, 5 lines |

#### 1.2.3 Meta Group (Right Column)

Wrapped in `gr.Group` with Markdown header "#### Meta".

| Component | Type | Default | Description |
|-----------|------|---------|-------------|
| `bpm` | Number | `None` | Beats per minute, precision=0 |
| `target_duration` | Number | `None` | Target duration in seconds, precision=0 |
| `key_scale` | Textbox | `""` | e.g., "C Major", "A minor" |
| `time_signature` | Textbox | `""` | e.g., "4/4", "3/4" |

Layout: Two rows with 2 components each.

#### 1.2.4 Config Accordion (Right Column - Collapsed by Default)

| Component | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `temperature` | Slider | 0.1 - 2.0 | 0.85 | Sampling temperature |
| `cfg_scale` | Slider | 1.0 - 5.0 | 2.0 | Classifier-free guidance scale |
| `top_k` | Number | - | `None` | Top-K sampling (0 or None to disable) |
| `top_p` | Slider | 0.0 - 1.0 | 0.9 | Top-P (nucleus) sampling |
| `repetition_penalty` | Slider | 1.0 - 2.0 | 1.0 | Repetition penalty |
| `metadata_temperature` | Slider | 0.1 - 2.0 | 0.85 | Temperature for metadata generation |
| `codes_temperature` | Slider | 0.1 - 2.0 | 1.0 | Temperature for codes generation |

#### 1.2.5 Generate Button

| Component | Type | Description |
|-----------|------|-------------|
| `generate_codes_btn` | Button (primary, large) | Label: "🎼 Generate Codes" |

**Handler Method:**
```python
handler.generate_llm_codes(
    caption: str,
    lyrics: str,
    temperature: float = 0.85,
    cfg_scale: float = 2.0,
    negative_prompt: str = "NO USER INPUT",
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    metadata_temperature: float = 0.85,
    codes_temperature: float = 1.0,
    target_duration: Optional[float] = None,
    user_metadata: Optional[Dict[str, str]] = None
) -> Tuple[Dict, str, str]  # (metadata, audio_codes, status)
```

**Implementation Details:**
- Calls `LLMHandler.generate_with_stop_condition()` with `infer_type="llm_dit"`
- Converts `top_k=0` to `None` (disabled)
- Converts `top_p>=1.0` to `None` (disabled)
- Builds `user_metadata` dict from bpm, keyscale, timesignature

---

### 1.3 Results Sub-Section

| Component | Type | Description |
|-----------|------|-------------|
| `metadata_output` | JSON | Generated metadata (read-only), scale=1 |
| `audio_codes_output` | Textbox | Generated audio codes, 8 lines, scale=2, `show_copy_button=True` |
| `llm_generation_status` | Textbox (read-only) | Generation status message |

---

## 2. ACEStep Section

The ACEStep Section generates audio from codes using the DiT model.

### 2.1 Model Loading Sub-Section

| Component | Type | Description |
|-----------|------|-------------|
| `dit_config_dropdown` | Dropdown | Available DiT configs from `handler.get_available_dit_models()`, scale=3 |
| `dit_device` | Dropdown | Options: `["auto", "cuda", "cpu"]`, default: `"auto"`, scale=1 |
| `load_dit_btn` | Button (primary) | Triggers `handler.initialize_dit()`, scale=1 |
| `dit_status` | Textbox (read-only) | Placeholder: "DiT model not loaded" |

**Handler Method:**
```python
handler.initialize_dit(config_path: str, device: str) -> str
```

**Implementation Details:**
- Auto-detects project root
- Calls `AceStepHandler.initialize_service()` with:
  - `use_flash_attention`: Auto-detected via `is_flash_attention_available()`
  - `compile_model`: False
  - `offload_to_cpu`: False

---

### 2.2 Task Type Sub-Section

| Component | Type | Options | Default |
|-----------|------|---------|---------|
| `task_type` | Dropdown | `["generate", "repaint", "cover", "add", "complete", "extract"]` | `"generate"` |

**Task Mapping (UI → Internal):**

| UI Task | Internal Type | Description |
|---------|---------------|-------------|
| `generate` | `text2music` | Generate music from text |
| `repaint` | `repaint` | Regenerate a portion of audio |
| `cover` | `cover` | Create a cover version |
| `add` | `lego` | Add a track to existing audio |
| `complete` | `complete` | Complete partial audio |
| `extract` | `extract` | Extract audio features |

---

### 2.3 Model Conditions Sub-Section (Dynamic Visibility)

Two-column layout:
- **Left Column (scale=1)**: Common inputs + Logical conditions
- **Right Column (scale=1)**: Dynamic groups + Advanced settings

#### 2.3.1 Common Inputs Group (All Tasks)

| Component | Type | Description |
|-----------|------|-------------|
| `ace_caption` | Textbox | Music description, 2 lines |
| `ace_lyrics` | Textbox | Lyrics, 3 lines |
| `ace_audio_codes` | Textbox | Audio codes, 3 lines |

#### 2.3.2 Logical Conditions Group (All Tasks)

| Component | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `inference_steps` | Slider | 1 - 100 | 20 | Number of diffusion steps |
| `guidance_scale` | Slider | 1.0 - 20.0 | 7.0 | CFG scale |
| `seed` | Number | - | -1 | Random seed (-1 for random) |
| `use_random_seed` | Checkbox | - | True | Use random seed |

#### 2.3.3 Meta Group (All Tasks)

Wrapped in `gr.Group` with Markdown header "#### Meta".

| Component | Type | Default | Description |
|-----------|------|---------|-------------|
| `ace_bpm` | Number | `None` | Beats per minute, precision=0 |
| `ace_target_duration` | Number | `None` | Target duration in seconds, precision=0 |
| `ace_key_scale` | Textbox | `""` | e.g., "C Major", "A minor" |
| `ace_time_signature` | Textbox | `""` | e.g., "4/4", "3/4" |

Layout: Two rows with 2 components each.

#### 2.3.4 Reference Audio Group (All Tasks)

**Visible for:** all tasks (generate, repaint, cover, add, complete, extract)

| Component | Type | Description |
|-----------|------|-------------|
| `reference_audio` | Audio (filepath) | Reference audio file for style guidance (optional) |

#### 2.3.5 Source Audio Group (Dynamic)

**Visible for:** repaint, cover, add, complete, extract

| Component | Type | Description |
|-----------|------|-------------|
| `source_audio` | Audio (filepath) | Source audio file to be processed (required for non-generate tasks) |

#### 2.3.6 Repaint Parameters Group (Dynamic)

**Visible for:** repaint

| Component | Type | Default | Description |
|-----------|------|---------|-------------|
| `repainting_start` | Number | 0.0 | Start time in seconds |
| `repainting_end` | Number | 10.0 | End time in seconds |

#### 2.3.7 Cover Parameters Group (Dynamic)

**Visible for:** cover

| Component | Type | Range | Default |
|-----------|------|-------|---------|
| `audio_cover_strength` | Slider | 0.0 - 1.0 | 1.0 |

#### 2.3.8 Track Parameters Group (Dynamic)

**Visible for:** add, complete

| Component | Type | Options | Default |
|-----------|------|---------|---------|
| `track_type` | Dropdown | `["vocal", "bass", "drums", "guitar", "piano", "other"]` | `"vocal"` |

#### 2.3.9 Advanced Settings Accordion (Collapsed by Default)

| Component | Type | Range | Default |
|-----------|------|-------|---------|
| `cfg_interval_start` | Slider | 0.0 - 1.0 | 0.0 |
| `cfg_interval_end` | Slider | 0.0 - 1.0 | 1.0 |
| `use_adg` | Checkbox | - | False |
| `use_tiled_decode` | Checkbox | - | True |
| `audio_format` | Dropdown | `["mp3", "wav", "flac"]` | `"mp3"` |
| `vocal_language` | Dropdown | `["en", "zh", "ja", "ko"]` | `"en"` |

---

### 2.4 Generate Button

| Component | Type | Description |
|-----------|------|-------------|
| `generate_audio_btn` | Button (primary, large) | Label: "🎵 Generate Audio" |

**Handler Method:**
```python
handler.generate_audio(
    task_type: str,
    caption: str,
    lyrics: str,
    audio_codes: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    reference_audio_path: Optional[str],
    repainting_start: float,
    repainting_end: float,
    audio_cover_strength: float,
    bpm: Optional[int],
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    use_adg: bool,
    cfg_interval_start: float,
    cfg_interval_end: float,
    audio_format: str,
    use_tiled_decode: bool,
    track_type: Optional[str] = None,
    progress=None
) -> Tuple[Optional[str], str, str]  # (audio_path, status, actual_texts)
```

**Implementation Details:**
- Maps UI task names to internal task names
- Generates instruction via `dit_handler.generate_instruction()`
- Calls `dit_handler.generate_music()` with all parameters
- Extracts audio path and status from result tuple

---

### 2.5 Results Sub-Section

| Component | Type | Description |
|-----------|------|-------------|
| `audio_output` | Audio (filepath) | Generated audio player |
| `actual_texts` | Textbox (read-only) | Actual text input |
| `audio_generation_status` | Textbox (read-only) | Status message |
---

## 3. Dynamic UI Logic

### 3.1 Task Visibility Configuration

```python
TASK_VISIBILITY = {
    "generate": {
        "reference_audio": True,
        "source_audio": False,
        "repaint_params": False,
        "cover_params": False,
        "track_params": False,
    },
    "repaint": {
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": True,
        "cover_params": False,
        "track_params": False,
    },
    "cover": {
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": True,
        "track_params": False,
    },
    "add": {
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": False,
        "track_params": True,
    },
    "complete": {
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": False,
        "track_params": True,
    },
    "extract": {
        "reference_audio": True,
        "source_audio": True,
        "repaint_params": False,
        "cover_params": False,
        "track_params": False,
    },
}
```

### 3.2 Visibility Update Function

```python
def update_task_visibility(task: str):
    """Update visibility of task-specific components based on selected task."""
    vis = TASK_VISIBILITY.get(task, TASK_VISIBILITY["generate"])
    return (
        gr.update(visible=vis["reference_audio"]),
        gr.update(visible=vis["source_audio"]),
        gr.update(visible=vis["repaint_params"]),
        gr.update(visible=vis["cover_params"]),
        gr.update(visible=vis["track_params"]),
    )
```

---

## 4. UI Layout Structure

```
gr.Blocks(title="ACE-Step Playground", theme=gr.themes.Soft())
├── gr.Markdown("# 🎵 ACE-Step Playground")
├── gr.Markdown("Generate music using LLM...")
│
└── gr.Tabs
    ├── gr.TabItem("🤖 LLM Section")
    │   ├── gr.Markdown("### Generate audio codes from text descriptions")
    │   │
    │   ├── gr.Accordion("1. Model Loading", open=True)
    │   │   ├── gr.Row [llm_model_dropdown, llm_backend, llm_device, load_llm_btn]
    │   │   └── llm_status
    │   │
    │   ├── gr.Accordion("2. Inputs", open=True)
    │   │   ├── gr.Row
    │   │   │   ├── gr.Column(scale=1)
    │   │   │   │   ├── caption
    │   │   │   │   ├── lyrics
    │   │   │   │   ├── negative_caption
    │   │   │   │   └── negative_lyrics
    │   │   │   └── gr.Column(scale=1)
    │   │   │       ├── gr.Group("#### Meta")
    │   │   │       │   ├── gr.Row [bpm, target_duration]
    │   │   │       │   └── gr.Row [key_scale, time_signature]
    │   │   │       └── gr.Accordion("Config", open=False)
    │   │   │           ├── gr.Row [temperature, cfg_scale]
    │   │   │           ├── gr.Row [top_k, top_p]
    │   │   │           ├── repetition_penalty
    │   │   │           └── gr.Row [metadata_temperature, codes_temperature]
    │   │   └── generate_codes_btn
    │   │
    │   └── gr.Accordion("3. Results", open=True)
    │       ├── gr.Row [metadata_output, audio_codes_output]
    │       └── llm_generation_status
    │
    └── gr.TabItem("🎹 ACEStep Section")
        ├── gr.Markdown("### Generate audio from codes using DiT model")
        │
        ├── gr.Accordion("1. Model Loading", open=True)
        │   ├── gr.Row [dit_config_dropdown, dit_device, load_dit_btn]
        │   └── dit_status
        │
        ├── gr.Accordion("2. Task & Conditions", open=True)
        │   ├── task_type
        │   └── gr.Row
        │       ├── gr.Column(scale=1)
        │       │   ├── gr.Group("#### Common Inputs")
        │       │   │   ├── ace_caption
        │       │   │   ├── ace_lyrics
        │       │   │   └── ace_audio_codes
        │       │   ├── gr.Group("#### Meta")
        │       │   │   ├── gr.Row [ace_bpm, ace_target_duration]
        │       │   │   └── gr.Row [ace_key_scale, ace_time_signature]
        │       │   └── gr.Group("#### Logical Conditions")
        │       │       ├── inference_steps
        │       │       ├── guidance_scale
        │       │       └── gr.Row [seed, use_random_seed]
        │       └── gr.Column(scale=1)
        │           ├── gr.Group("#### Reference Audio")
        │           │   └── reference_audio
        │           ├── gr.Group("#### Source Audio", visible=dynamic)
        │           │   └── source_audio
        │           ├── gr.Group("#### Repaint Parameters", visible=dynamic)
        │           │   └── gr.Row [repainting_start, repainting_end]
        │           ├── gr.Group("#### Cover Parameters", visible=dynamic)
        │           │   └── audio_cover_strength
        │           ├── gr.Group("#### Track Parameters", visible=dynamic)
        │           │   └── track_type
        │           └── gr.Accordion("Advanced Settings", open=False)
        │               ├── gr.Row [cfg_interval_start, cfg_interval_end]
        │               ├── gr.Row [use_adg, use_tiled_decode]
        │               └── gr.Row [audio_format, vocal_language]
        │
        ├── generate_audio_btn
        │
        └── gr.Accordion("3. Results", open=True)
            ├── audio_output
            ├── actual_texts
            └── audio_generation_status
```

---

## 5. Wrapper Functions

### 5.1 LLM Generate Codes Wrapper

```python
def generate_codes_wrapper(
    caption, lyrics, negative_caption, negative_lyrics,
    bpm, key_scale, time_signature, target_duration,
    temperature, cfg_scale, top_k, top_p,
    repetition_penalty, metadata_temperature, codes_temperature
):
    """Wrapper function to prepare inputs and call handler."""
    negative_prompt = negative_caption if negative_caption else "NO USER INPUT"
    
    user_metadata = {}
    if bpm is not None:
        user_metadata["bpm"] = str(int(bpm))
    if key_scale:
        user_metadata["keyscale"] = key_scale
    if time_signature:
        user_metadata["timesignature"] = time_signature
    
    return handler.generate_llm_codes(
        caption=caption,
        lyrics=lyrics,
        temperature=temperature,
        cfg_scale=cfg_scale,
        negative_prompt=negative_prompt,
        top_k=int(top_k) if top_k is not None else None,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        metadata_temperature=metadata_temperature,
        codes_temperature=codes_temperature,
        target_duration=target_duration,
        user_metadata=user_metadata if user_metadata else None
    )
```

### 5.2 ACEStep Generate Audio Wrapper

```python
def generate_audio_wrapper(
    task, caption, lyrics, codes,
    steps, guidance, seed_val, random_seed,
    ref_audio, repaint_start, repaint_end, cover_strength,
    track_type_val,
    bpm_val, key_val, time_sig_val, vocal_lang,
    adg, cfg_start, cfg_end, fmt, tiled,
    progress=gr.Progress(track_tqdm=True)
):
    """Wrapper function to call handler."""
    actual_seed = -1 if random_seed else int(seed_val) if seed_val is not None else -1
    
    return handler.generate_audio(
        task_type=task,
        caption=caption,
        lyrics=lyrics,
        audio_codes=codes,
        inference_steps=int(steps),
        guidance_scale=guidance,
        seed=actual_seed,
        reference_audio_path=ref_audio,
        repainting_start=repaint_start,
        repainting_end=repaint_end,
        audio_cover_strength=cover_strength,
        bpm=int(bpm_val) if bpm_val is not None else None,
        key_scale=key_val or "",
        time_signature=time_sig_val or "",
        vocal_language=vocal_lang,
        use_adg=adg,
        cfg_interval_start=cfg_start,
        cfg_interval_end=cfg_end,
        audio_format=fmt,
        use_tiled_decode=tiled,
        track_type=track_type_val,
        progress=progress
    )
```

---

## 6. Error Handling

### 6.1 Model Not Loaded Checks

```python
# In handler.generate_llm_codes()
if not self.llm_handler.llm_initialized:
    return {}, "", "❌ LLM not initialized. Please load the LLM model first."

# In handler.generate_audio()
if self.dit_handler.model is None:
    return None, "❌ DiT model not initialized. Please load the DiT model first."
```

### 6.2 Exception Handling

```python
try:
    # Generation logic
except Exception as e:
    return {}, "", f"❌ Error generating codes: {str(e)}\n{traceback.format_exc()}"
```

---

## 7. Dependencies

- gradio >= 4.0
- torch
- acestep modules (handler, llm_inference)

---

## 8. Running the Playground

```bash
# Basic
python playground/playground.py

# With options
python playground/playground.py --port 7860 --listen --share
```

---

## 9. Implementation Checklist

- [x] Create `playground_handler.py` with proper integration to `llm_inference.py` and `handler.py`
- [x] Create `playground_ui.py` with Tabs layout
- [x] Implement LLM Section with 3 sub-sections
- [x] Implement ACEStep Section with model loading, task selection, conditions, and results
- [x] Add dynamic visibility for task-specific components
- [x] Add Config/Advanced accordions (collapsed by default)
- [x] Add `track_type` parameter for add/complete tasks
- [x] Create `playground.py` entry point
- [ ] Test all task types
- [ ] Test dynamic UI visibility
- [ ] Test error handling
- [ ] Add progress tracking for generation
