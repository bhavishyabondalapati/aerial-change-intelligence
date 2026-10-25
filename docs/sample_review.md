# Report validation

Numbers and dates checked: 40 verified against the facts, 0 found in cited guides, **0 mismatched**, **2 unsupported**.
Sentences flagged for human review: **11** of 33.

| # | Sentence | Why it needs review |
|---|---|---|
| 1 | Satellite NDVI drops serve as signals for possible vegetation changes or stress, not confirmed crop stress. | overclaiming word: confirmed |
| 2 | The building change detection model was evaluated over 128 images using a threshold of 0.5. | low-confidence result (vegetation_loss_two_dates confidence 0.48) |
| 3 | Two-date loss (2024-12-30 to 2025-03-15): Flagged 3,356 regions covering 49,252,100.0 m² (4,925.21 ha) with a mean confidence of 0.48. | low-confidence result (vegetation_loss_two_dates confidence 0.48) |
| 4 | The maximum region area was 8,480,100.0 m². | low-confidence result (vegetation_loss_two_dates confidence 0.48) |
| 5 | Parameters included a drop threshold of 0.15, a cloud buffer of 50 m, and a valid fraction of 0.68. | low-confidence result (seasonal_anomaly confidence 0.68, vegetation_loss_two_dates confidence 0.48) |
| 6 | Seasonal anomaly (2025 vs. 2021–2024 baseline, window 02-01..03-10): Flagged 5,997 regions totaling 30,051,500.0 m² (3,005.15 ha) with a mean confidence of 0.68. | numbers not found in the facts or cited sources: 01, 10; low-confidence result (seasonal_anomaly confidence 0.68, vegetation_loss_two_dates confidence 0.48) |
| 7 | The maximum region area was 1,521,700.0 m². | low-confidence result (seasonal_anomaly confidence 0.68) |
| 8 | Out of a comparable area of 22,459.22 ha, the flagged fraction was 0.15 (0.1464). | low-confidence result (seasonal_anomaly confidence 0.68, vegetation_loss_two_dates confidence 0.48) |
| 9 | Target mean NDVI was 0.59 compared to the normal baseline mean of 0.66. | low-confidence result (seasonal_anomaly confidence 0.68) |
| 10 | Field officers should conduct physical inspections to confirm crop status: | overclaiming word: confirm |
| 11 | Satellite NDVI drops are observational signals only and do not provide proof of confirmed crop stress. | overclaiming word: confirmed |
