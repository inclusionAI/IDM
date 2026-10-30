# IDM project page

A static, bilingual guide to *Intrinsic Distribution Matching*.

Preview from the repository root:

```sh
python3 -m http.server 8765 --directory docs
```

Open `http://localhost:8765/?lang=en` or `http://localhost:8765/?lang=zh`.
The page also works directly from `index.html`. No build step or external
font, script, or formula service is required.

For GitHub Pages, serve the `main` branch's `/docs` directory as the site root.
The configured project URL is `https://inclusionai.github.io/IDM/`.
`assets/social-cover.png` is the prepared preview image.

## Content and assets

- All ten equation definitions in `equations.js` reproduce the original
  manuscript's LaTeX, with labels removed. Equation numbers match the paper.
- The hero uses four native 1024×1024 **IDM+ supplementary samples**, from
  the Qwen-Image-2.1 / VGG-MMD extension at 2 NFE: Chinese bakery chalkboard,
  graffiti, concert poster, and portrait. These are supplementary visual
  examples; the quantitative results on this page remain those of the
  original IDM paper. The extension uses VGG features.
- Hero WebP previews use high-quality compression. Clicking loads the
  unmodified original PNG. Original image hashes and source case IDs are in
  `assets/samples-provenance.json`.
- Earlier Figure 2 book/portrait and Figure 5 coffee/sushi crops are retained
  in the assets directory as original-paper references.
- `decomposition.webp`: paper Figure 3, rendered from `cal.pdf`.
- `ablation.webp`: the original paper Figure 5, retained with its labels.
- `social-cover.png`: selected supplementary samples and original-paper
  results, with their respective models explicitly labeled. Sample contents
  have not been retouched or regenerated.
- `inclusion-ai.png`: the logo used by the original paper.
- Results come from paper Table 5. The interactive density curves are
  illustrative one-dimensional Gaussian proxies, not measured features.

The page keeps external method comparisons separate from controlled
ablations. Checkpoint availability remains **coming soon**.

KaTeX 0.16.22 is bundled under its MIT license; see
`vendor/katex/LICENSE`. Figures and samples are from the IDM paper.
