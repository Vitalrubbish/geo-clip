# Hugging Face Processor Timeout Fix

## Root Cause

The baseline reproduction was timing out on a `HEAD` request to
`https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/processor_config.json`
because `geoclip/model/image_encoder.py` always instantiated the image
preprocessor with `AutoProcessor.from_pretrained(...)`.

That API reaches out to Hugging Face to resolve processor metadata, so even
after the GeoCLIP weights finished loading, the model still tried to contact the
Hub for `processor_config.json`. In restricted or flaky network environments,
that extra metadata fetch fails with a timeout.

## Fix

`ImageEncoder` now loads the CLIP image processor in a cache-first way:

1. Try `CLIPImageProcessor.from_pretrained(..., local_files_only=True)`.
2. If the processor files are not cached locally, fall back to a pure
   torchvision preprocessing pipeline that matches CLIP's expected resize,
   crop, tensor conversion, and normalization.

This removes the dependency on the online processor metadata request and keeps
baseline evaluation runnable in offline or high-latency environments.

## Impact

- The pretrained CLIP backbone is unchanged.
- Image preprocessing still matches CLIP conventions.
- The evaluation script no longer blocks on the Hugging Face processor HEAD
  request.