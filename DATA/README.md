# Example Input

`D0064/` contains 34 PNG images and 34 paired JSON annotations. Image pixels are unchanged from the existing input set. JSON tooth polygons and segmentation labels are retained; `imagePath` points to the paired PNG and `imageData` is null so the external image is used.

The first 32 slots follow this FDI order:

```text
18 17 16 15 14 13 12 11
21 22 23 24 25 26 27 28
38 37 36 35 34 33 32 31
41 42 43 44 45 46 47 48
```

Slot 33 is the whole lower-jaw view; slot 34 is the whole upper-jaw view. `selected_34_manifest.csv` records the target tooth, actual substitute tooth and view-angle rule for each slot. It contains no absolute source paths.

The 34 slots do not imply that the patient has 32 natural teeth. Some slots use neighboring-tooth substitute views, and different slots may reuse a view. Do not infer tooth presence from a filename or treat a substitute tooth as the intended tooth's annotation. The visible tooth identities are defined by `shapes[].label` in each JSON file.

These labels identify segmented teeth; they are not clinical periodontitis ground truth. The example contains neither clinical diagnoses nor fabricated predictions. It demonstrates the existing 34-view interface input format, not a full multi-angle dataset.
