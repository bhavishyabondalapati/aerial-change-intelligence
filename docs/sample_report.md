# Change intelligence report: Godavari delta

## Summary
This report outlines building change detection metrics and satellite-based vegetation monitoring results for Tile 44QPD in the Godavari delta. Satellite NDVI drops serve as signals for possible vegetation changes or stress, not confirmed crop stress.

## Building change detection
The building change detection model was evaluated over 128 images using a threshold of 0.5. The test metrics are:
- Precision: 0.91 (0.9077)
- Recall: 0.89 (0.8885)
- F1 score: 0.90 (0.8980)
- IoU: 0.81 (0.8149)
- Expected Calibration Error (ECE): 0.0066

For a building change example, 101 regions were identified across 27,479.5 m² (2.75 ha), with a mean confidence of 0.97 and a maximum single region area of 767.75 m².

## Vegetation change in the Godavari delta
Two satellite vegetation analyses were recorded:
1. **Two-date loss (2024-12-30 to 2025-03-15):** Flagged 3,356 regions covering 49,252,100.0 m² (4,925.21 ha) with a mean confidence of 0.48. The maximum region area was 8,480,100.0 m². Parameters included a drop threshold of 0.15, a cloud buffer of 50 m, and a valid fraction of 0.68.
2. **Seasonal anomaly (2025 vs. 2021–2024 baseline, window 02-01..03-10):** Flagged 5,997 regions totaling 30,051,500.0 m² (3,005.15 ha) with a mean confidence of 0.68. The maximum region area was 1,521,700.0 m². Out of a comparable area of 22,459.22 ha, the flagged fraction was 0.15 (0.1464). Target mean NDVI was 0.59 compared to the normal baseline mean of 0.66.

## Possible causes
Satellite NDVI drops indicate possible changes in vegetation condition. Specific ground causes for these drops are unknown without direct field checks. Based on regional guidelines, possible agronomic causes include:
- **Nutrient Deficiencies:** Deficiencies in nitrogen, phosphorus, potassium, zinc, or iron can cause leaf yellowing, chlorosis, stunting, or tissue drying [S1, S8, S9].
- **Water & Cropping Management:** Delayed canal water release, shifting from paddy to pulses, or altered irrigation schedules (such as alternate wetting and drying) [S4, S5, S6].
- **Biotic and Abiotic Stresses:** Adverse weather factors or pests and diseases such as brown planthopper, bacterial blight, or viruses [S3, S8].

## Recommended field checks
Field officers should conduct physical inspections to confirm crop status:
- **Visual Diagnosis:** Check affected fields for stunting, chlorosis, necrosis, or wilting [S11, S12]. Compare affected plants against healthy plants of the same variety and age [S11, S12].
- **Field & Management Review:** Examine drainage conditions, fertilization history, and consult farmers regarding management practices and inputs [S11, S12].
- **Nutrient Testing:** Verify visual symptoms using soil tests or biological methods like the Minus-One Element Technique [S13]. Take leaf samples for tissue analysis after irrigation or rainfall to verify nutrient uptake [S13].

## Limitations
- Satellite NDVI drops are observational signals only and do not provide proof of confirmed crop stress.
- Confidence values represent scaled NDVI drops and are not statistical probabilities, as no ground truth is available for the crop tracks.
- Specific causes for vegetation changes in individual fields remain unknown until verified on the ground.

## Sources

- **[S1]** rice_disorders_field_guide_philrice.pdf, page 34
- **[S2]** rice_disorders_field_guide_philrice.pdf, page 39
- **[S3]** rice_disorders_field_guide_philrice.pdf, page 40
- **[S4]** east_godavari_contingency_plan_icar_crida.pdf, page 13
- **[S5]** east_godavari_contingency_plan_icar_crida.pdf, page 15
- **[S6]** east_godavari_contingency_plan_icar_crida.pdf, page 13
- **[S7]** east_godavari_contingency_plan_icar_crida.pdf, page 10
- **[S8]** rice_disorders_field_guide_philrice.pdf, page 5
- **[S9]** rice_disorders_field_guide_philrice.pdf, page 35
- **[S10]** rice_disorders_field_guide_philrice.pdf, page 7
- **[S11]** rice_disorders_field_guide_philrice.pdf, page 10
- **[S12]** rice_disorders_field_guide_philrice.pdf, page 10
- **[S13]** rice_disorders_field_guide_philrice.pdf, page 36
- **[S14]** rice_disorders_field_guide_philrice.pdf, page 9
- **[S15]** east_godavari_contingency_plan_icar_crida.pdf, page 41
- **[S16]** east_godavari_contingency_plan_icar_crida.pdf, page 43
