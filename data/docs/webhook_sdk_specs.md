# Webhook & SDK Specifications (SOP-105)
- **Dropped Nested JSON Metadata**: Webhook delivery payloads are capped at 256 KB. Payloads exceeding 256 KB have nested `metadata` objects truncated. Store large payloads in object storage and pass a signed URL instead.
- **Webhook Signature Verification**: Every request includes an `X-Aegis-Signature` HMAC-SHA256 header computed over `timestamp + "." + raw_body`. Reject requests older than 300 seconds to prevent replay attacks.
