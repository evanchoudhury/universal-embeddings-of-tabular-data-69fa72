Direction: Eclipse (warm graphite night, one ember corona, smoked glass, editorial serif + mono readouts)

## Sources
- https://www.digidop.com/blog/5-web-design-trends-2025/ : 2026 trend "light, materials and shaders". The 2025 glow look is changing; use one light or material effect per page, placed on the element you want noticed. Big page-wide background gradients are fading.
- https://dev.to/thekitbase/saas-dashboard-design-trends-in-2026-with-examples-121m : a move toward dark-first designs with minimal chrome and denser data, because dark backgrounds lift chart contrast.
- https://925studios.co/blog/saas-dashboard-design-examples-2026 : progressive disclosure is the key 2026 dashboard pattern. That's why the sticky tab bar and expanders get the strongest treatment.
- https://www.awwwards.com/websites/data-visualization : award sites pair editorial display faces (Editorial New, Instrument-style serifs) with mono faces (Fragment Mono-style) for data.
- UI/UX Pro Max: style #51 HUD / Sci-Fi FUI (fine lines, alert red, mono readouts), style #14 Liquid Glass (glass for navigation and controls only, never on content). Typography rows #59/#60 (serif + mono stacks).

## What is new compared with the journal
- **Warm instead of cool, dark instead of light.** Orchid was a light pink-violet page. Lagoon was a dark cool cyan/green. Eclipse is a warm near-black (#0C0B0E) with a single amber corona above the HUD. The amber is the app's own `>50K` data colour (#d95f02) made lighter, so the Plotly charts, whose colours the theme can't change, now sit inside the palette.
- **First serif voice.** Headings use Instrument Serif (editorial, 400 weight, large). Earlier projects used geometric sans (DM Sans, Outfit). Body text is Geist. Geist Mono is used only for measurements: metric values with tabular figures, deltas and code tokens. h4 section labels are small mono caps in amber.
- **Glass has a rule this time.** Content panels are *smoked* glass: a warm tint, a glossy top highlight, a faint warm glint from the corona's side and a real offset shadow, with no blur because nothing scrolls behind them. Real backdrop blur is used only where content does scroll underneath: the header and a new **sticky frosted tab bar**. The selected tab glows from below like a lit lens.
- **Film grain.** A 4.5%-alpha SVG noise layer, URL-encoded inline so the CSS stays free of the `<` character. It stops the dark page looking like flat digital black.
- **Pipeline graph on lit paper.** The graphviz chart (dark ink on a transparent background) gets a warm paper card, like a slide in a dark room. Without it, black edges would vanish on the dark page.
- **HUD:** dots streaks (first time), 128 thin segments with a wide 0.44 gap (fine ticks, not chunky bars), counter-rotation (spin -1.3), and rings amber / mint / champagne. Baseline is the champagne ring, so the centre core glows warm like a corona. Alarm red is #FF4D63, far from amber in hue, so wrong predictions read as alarms.
- Browser surfaces are themed: amber text selection and caret, amber focus ring, thin warm scrollbars, link underline offset, tag chips, and a dashed amber upload dropzone.

## Lessons for the next project
1. Fix the colour of what you can't style first. The app's hard-coded chart colours (orange/green) should choose the page palette, not fight it. Check app.py for literal hex values before picking a hue.
2. Give glass a reason. Blur only where content actually passes behind (sticky bars, header); everywhere else a tint plus specular edge plus offset shadow reads as glossy for less GPU cost and fewer contrast surprises.
3. Look for third-party renders that ignore the theme (graphviz, PNG plots, st.bar_chart). Give them a deliberate surface, like a paper card or padded frame, rather than hoping they show up on a dark page.
