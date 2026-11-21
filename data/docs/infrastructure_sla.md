# Infrastructure, Memory Leaks & Data Residency (SOP-104)
- **Worker Container Memory Leak**: Verified bug in worker image `v1.8.2` where async HTTP connection pools are not closed after 12 hours of uptime. Upgrade to `v1.8.4` or configure Kubernetes `maxRequestsPerChild=5000`.
- **Vector Index Sync Stuck at 99%**: Happens when a background HNSW graph compaction locks during high write throughput. Pause ingestion for 3 minutes to let the compaction lock release automatically.
- **EU-Central Frankfurt Data Residency**: Enterprise and Pro workspaces can pin all vector storage and inference logs to `eu-central-1` (Frankfurt) for GDPR compliance. Migration requires a 10-minute read-only maintenance window.
