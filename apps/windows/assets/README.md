# NInfer application icon

`ninfer.png` is the source artwork. `ninfer.ico` contains 16, 20, 24, 32, 40, 48,
64, 128, and 256 pixel frames and is embedded in the Windows Supervisor, server,
and CLI executables. The tray keeps its separate status-colored indicator.

Rebuild the ICO after replacing the artwork:

```powershell
.\scripts\windows\build-icon.ps1
```

Artwork created with the built-in image generation tool. The final edit uses an
opaque navy background because the initial output painted a checkerboard instead
of providing an alpha channel.

## Generation prompt

Use case: logo-brand. Asset type: a single finished Windows desktop application icon for NInfer, a fast local GPU inference engine. Create a beautiful, distinctive, bold geometric capital N monogram, built from two strong upright strokes connected by a swift diagonal, with subtle faceted depth suggesting computation and speed. Clean contemporary desktop-app craftsmanship, luminous cyan and turquoise with restrained blue shading, on a very dark midnight rounded-square tile. Large simple silhouette, generous stroke thickness, instantly readable at 16, 32 and 48 pixels. The N occupies most of the tile with balanced padding. Front view, exactly square canvas, tile almost fills the canvas with a very narrow transparent margin outside its rounded corners. Actual transparent background outside the tile. Crisp edges, restrained soft highlights, no glow spilling outside the silhouette. Output only the one icon, not a presentation sheet, no mockup, no surrounding scenery, no captions or wordmark, no tiny circuit lines, no brains, no sparkles, no watermark. Only the single N symbol.

## Final edit prompt

Edit target: the supplied NInfer icon. Keep the beautiful faceted cyan N monogram, its geometry, dimensions and shading unchanged. Make this a FULL-BLEED SQUARE icon: extend the same very dark midnight navy tile background to cover every single pixel of the entire square canvas including all four corners and outer edges. Remove the gray checkerboard everywhere, replacing it with the continuous navy background. No transparency is wanted now. Remove the rounded tile rim and border so the N sits on a clean uninterrupted full-bleed navy field. Output one finished square Windows application icon. No checkerboard, no outer margins, no mockup, no text other than the N, no new elements.
