# API & Inference Error Runbook (SOP-101)
- **HTTP 502 Bad Gateway on Inference Endpoint**: Caused by worker container OOM (Out-Of-Memory) when batch size exceeds 64 or embedding payload exceeds 8,192 tokens. Immediate mitigation: restart the inference deployment pod and enforce client-side chunking at 4,096 tokens.
- **TimeoutError in Python SDK**: Default client timeout is 15 seconds. For batch sizes between 32 and 64, set `timeout=60.0` and enable exponential backoff retries (`max_retries=3`).
- **NaN Values in Embeddings Endpoint**: Occurs when an empty string `""` or whitespace-only string is passed in the `input` array. Filter empty strings prior to API serialization.
- **Negative Rate Limit Headers**: Known display bug in API Gateway v2.4 when sliding-window Redis counters reset; does not block actual requests unless HTTP 429 is returned.
