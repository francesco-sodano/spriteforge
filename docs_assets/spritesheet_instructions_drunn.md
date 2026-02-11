# Spritesheet Generation Instructions — Drunn Ironhelm (Berserker)

> **Workflow:** Generate the **Base Character Image** first to establish the design. Then generate each animation row **one at a time** as a separate horizontal sprite strip. Finally, assemble all 16 rows into the final 896×1024 spritesheet.

---

## Global Technical Rules (apply to ALL prompts below)

- **Art style:** Modern HD pixel art (detailed, clean pixel art at 2× density — like Dead Cells or Owlboy). NOT low-res NES-style.
- **Frame size:** Each frame is exactly **64×64 pixels**.
- **Background:** Fully **transparent** (alpha channel). Output as **PNG-32**.
- **No anti-aliasing to background** — crisp pixel-art edges against transparency.
- **Consistent 1-pixel dark outline** (dark brown or black) around the character in every frame.
- **Facing direction:** Character faces **RIGHT** in all frames.
- **Feet anchor:** Feet at approximately **y=56** within each 64×64 frame (8px from bottom) for ground alignment.
- **Character centered** horizontally within each 64×64 cell. Axe swings may extend near edges but must not be clipped.
- **Unused cells** in a row must be fully transparent/empty.
- **No border/padding** between frames — tightly packed in the 64×64 grid.
- **IMPORTANT — Drunn is the shortest but WIDEST character.** He is a stocky dwarf: ~42–46 pixels tall but ~36–40 pixels wide. Proportions must be squat and thick, NOT tall and lean.

---

## Prompt 0: Base Character Image

> **Purpose:** Generate a single **character reference sheet** to lock down the visual design before creating animation frames. This is NOT a sprite — it's a high-detail reference at a larger size.

**Prompt:**

Generate a pixel art character reference sheet for a fantasy dwarven berserker named **Drunn Ironhelm**. Modern HD pixel art style (like Dead Cells / Owlboy). The sheet must follow the exact layout described below.

**Reference Sheet Layout — 1024×768 pixel canvas:**

The image must follow this exact layout so all three character sheets are visually consistent:

- **Background:** Solid blue (R=0, G=0, B=255) across the entire canvas. No gradients, no patterns.
- **Top banner (y: 0–64):**
  - Character name **"DRUNN IRONHELM"** in large **white (#FFFFFF) pixel-art uppercase text**, centered horizontally at y≈20.
  - Class label **"BERSERKER"** in smaller white text, centered below the name at y≈46.
- **Left panel — FRONT VIEW (x: 40–490, y: 80–620):**
  - Full-body neutral standing pose facing the viewer (front-facing), centered within the panel.
  - Character drawn at approximately **4× sprite scale** (~168–184px tall, ~144–160px wide) for detail visibility. Note: Drunn is much shorter and wider than the other characters — squat dwarven proportions.
  - Twin axes at sides, horned helm, braided beard clearly visible.
  - White label text **"FRONT"** centered below the character at y≈630.
- **Right panel — 3/4 SIDE VIEW (x: 534–984, y: 80–620):**
  - Same full-body neutral standing pose, rotated to a **3/4 right-facing perspective** (matching the in-game sprite facing direction).
  - Same 4× scale as the front view. Horned helm profile, massive pauldrons, and twin axes clearly visible from this angle.
  - White label text **"3/4 SIDE"** centered below the character at y≈630.
- **Bottom strip — COLOR PALETTE (y: 660–740):**
  - A horizontal row of color swatches, evenly spaced and centered across the canvas width.
  - Each swatch is a **32×32 pixel filled square** with a **1px white border**.
  - Below each swatch: the element name in small white uppercase text (e.g., "SKIN", "BEARD", "HELM").
  - Show **P1 palette only** in swatch order matching the P1 table below (left-to-right: Skin, Beard, Helm, Pauldrons, Chainmail, Leather, Boots, Axe heads, Axe hafts, Red accent).

**Character description:**
- **Build:** Stocky, barrel-chested, massively muscular dwarf. The shortest of the three playable characters but by far the widest and heaviest. ~42–46 pixels tall in-game sprite, ~36–40 pixels wide. Think fire hydrant proportions — low center of gravity, immovable. Arms are thick as tree trunks, nearly touching the ground.
- **Skin:** Ruddy tan, weathered and scarred — RGB approximately (190, 145, 110). Visible scars on exposed arms.
- **Beard:** Long, fiery red-orange (180, 70, 20), braided into two thick braids that reach mid-chest. Iron rings/clasps woven into the braids. Beard is one of his defining visual features.
- **Hair:** Same fiery red, mostly hidden beneath the helm. What's visible is shaggy and wild at the sides.
- **Face:** Broad, flat nose. Bushy red eyebrows. Small, fierce eyes (dark amber). Permanent scowl or battle-fury expression.
- **Helm:** Bronze/copper (170, 120, 50) Viking-style horned helmet. Two curved ram-like horns extending from the sides. Helmet covers the top of the head, leaves the face and beard exposed. Riveted with darker bronze studs. This helm is ALWAYS worn — it is his signature.
- **Armor — heavy plate and chainmail:**
  - Thick bronze/copper (170, 120, 50) breastplate over chainmail. The breastplate has a geometric dwarven rune etched into the center.
  - Heavy bronze (160, 110, 45) pauldrons (oversized shoulder guards) — nearly as wide as his head, with spike/stud details.
  - Chainmail sleeves (130, 130, 140) visible from pauldron to elbow.
  - Bare forearms below the elbow — massive, scarred, with leather wrist-wraps.
  - Wide leather (90, 60, 35) war-belt with large iron buckle.
  - Chainmail skirt over heavy leather pants, ending at the knee.
  - Thick-soled dark iron/dark brown (60, 55, 50) heavy boots. Wide, flat feet.
  - Red accent details (170, 35, 30): a knotwork pattern on belt, trim on pauldrons. This is the key P2 swap color.
- **Weapons:**
  - **Twin Heavy War Axes:** One in each hand. Broad, single-bladed crescent heads of dark steel (100, 100, 110) with leather-wrapped (120, 80, 45) wooden hafts. Each axe ~22–26 pixels long. Hefted at his sides or resting on shoulders.
  - These axes are ALWAYS in his hands (unlike Sylara who switches weapons).
- **Distinctive silhouette features:** Extremely wide and short, horned helm, massive pauldrons, twin axes, braided beard. Should look like a walking widget — wider than he is tall by proportion.

**P1 Color Palette:**

| Element | RGB |
|---|---|
| Skin | (190, 145, 110) |
| Beard/hair | (180, 70, 20) |
| Helm/breastplate | (170, 120, 50) |
| Pauldrons | (160, 110, 45) |
| Chainmail | (130, 130, 140) |
| Leather belt/pants | (90, 60, 35) |
| Boots | (60, 55, 50) |
| Axe heads (steel) | (100, 100, 110) |
| Axe hafts (wood) | (120, 80, 45) |
| Red accent | (170, 35, 30) |

**P2 Palette (alternate — for palette swap):**

| Element | P1 → P2 |
|---|---|
| Helm/breastplate/pauldrons | (170, 120, 50) → (175, 180, 190) |
| Horns | (170, 120, 50) → (155, 160, 170) |
| Red accent | (170, 35, 30) → (40, 70, 160) |

P2 must be **exact flat color swaps** — no gradients. The game engine does pixel-level color replacement.

**Output:** `drunn_base_reference.png` — **1024×768 pixels**. PNG with solid blue background (0, 0, 255).

---

## Prompt 1: IDLE — Row 0 (6 frames, looping)

Generate a **horizontal sprite strip** of **6 frames**, each frame **64×64 pixels**, for a total image size of **384×64 pixels**. PNG-32 with transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm — stocky dwarven berserker with bronze horned helm, red braided beard, twin heavy axes. Very short (~42–46px) and very wide (~36–40px). Faces RIGHT.

**Animation:** Impatient, aggressive idle — this is a berserker, not a patient man.
- Both axes held at his sides, gripped tightly.
- Frame 0–5: Heavy breathing — large chest heaving. Occasional restless foot stomp or knuckle crack. Beard sways with breath. Head may tilt side to side.
- Loops seamlessly (frame 5 → frame 0).
- Feet at y=56. 1px dark outline. Note Drunn's feet are at y=56 but his head reaches only ~y=10–14 due to short stature (with horn tips reaching ~y=6–8).

**Frame timing:** 160ms per frame (slowest idle — heavy, ponderous).

**Output:** `drunn_row00_idle.png` — 384×64 pixels.

---

## Prompt 2: WALK — Row 1 (8 frames, looping)

Generate a **horizontal sprite strip** of **8 frames**, each **64×64 pixels**, total **512×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Stomping, heavy walk — the ground should practically shake. Slow but unstoppable.
- Short-legged stride, body stays very low. Almost no vertical movement — too heavy.
- Both axes swing slightly with each step. Beard sways. Pauldrons shift.
- Heavy boot impacts — stomping gait. Deliberate.
- Seamless 8-frame walk loop. Feet at y=56. 1px dark outline.

**Frame timing:** 120ms per frame.

**Output:** `drunn_row01_walk.png` — 512×64 pixels.

---

## Prompt 3: ATTACK1 — Row 2 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. Twin heavy war axes.

**Animation:** RIGHT axe overhead chop — brutal, heavy downward strike. First hit of a 3-hit combo.
- Frame 0: Wind-up — right axe reared back over shoulder. Body leans back from the weight.
- Frame 1: Swinging down — axe arcing overhead.
- Frame 2: **Active hit frame** — right axe slams down. Extended reach ~18px past body. Impact frame.
- Frame 3: Axe embedded in ground momentarily — body leaned into the strike.
- Frame 4: Wrenching axe free, recovery.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame (slower than other characters — heavier weapons).

**Output:** `drunn_row02_attack1.png` — 320×64 pixels.

---

## Prompt 4: ATTACK2 — Row 3 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. Twin heavy war axes.

**Animation:** LEFT axe horizontal sweep — wide arc at torso height. Second hit of combo. Mirror-direction from ATTACK1.
- Frame 0: Left axe pulled back across body.
- Frame 1: Torso twisting — rotational momentum building.
- Frame 2: **Active hit frame** — left axe sweeps in a wide horizontal arc. Maximum reach.
- Frame 3: Follow-through — body continuing rotation.
- Frame 4: Recovery — both axes regrouping.
- Feet at y=56. 1px dark outline.

**Frame timing:** 90ms per frame.

**Output:** `drunn_row03_attack2.png` — 320×64 pixels.

---

## Prompt 5: ATTACK3 — Row 4 (7 frames, one-shot)

Generate a **horizontal sprite strip** of **7 frames**, each **64×64 pixels**, total **448×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. Twin heavy war axes.

**Animation:** BOTH axes double-overhead slam — devastating combo finisher. Both axes crash down simultaneously. Most damaging attack (Strength 10/10). Causes enemy knockdown.
- Frame 0: Crouching — coiling power.
- Frame 1: Both axes hoisted high overhead — body stretched to full (short) height. Massive windup.
- Frame 2: Leaping slightly off ground — putting full body weight into the strike.
- Frame 3: **Active hit frame** — both axes slamming down simultaneously. Maximum impact.
- Frame 4: Impact — shockwave pose, body fully compressed from the follow-through.
- Frame 5: Wrenching both axes from the ground.
- Frame 6: Recovery — standing back up, breathing heavily.
- Feet at y=56. 1px dark outline.

**Frame timing:** 110ms per frame.

**Output:** `drunn_row04_attack3.png` — 448×64 pixels.

---

## Prompt 6: JUMP — Row 5 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. Both axes in hand.

**Animation:** Stubby, labored jump — Drunn is Heavy. This is NOT a graceful jump. He barely gets off the ground compared to other characters.
- Frame 0: Crouch — deep squat, straining under his own weight. Knees bent.
- Frame 1: Launch — pushing off with massive legs. Body barely lifts.
- Frame 2: Apex — just barely airborne. Arms spread for balance. Short hang time.
- Frame 3: Descent — crashing back down, legs bracing for heavy landing.
- Feet at y=56 in frames 0 and 3. 1px dark outline.

**Frame timing:** 110ms per frame.

**Output:** `drunn_row05_jump.png` — 256×64 pixels.

---

## Prompt 7: JUMP_ATTACK — Row 6 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. Twin axes.

**Animation:** Aerial axe slam — body-slam style. Drunn brings both axes down while dropping like a boulder.
- Frame 0: Airborne — rising, both axes above head.
- Frame 1: Apex — axes cocked overhead, body curling forward.
- Frame 2: **Active hit frame** — plummeting down, both axes swinging downward. Full body weight committed.
- Frame 3: Landing — smashing into ground, heavy impact crouch.
- 1px dark outline.

**Frame timing:** 90ms per frame.

**Output:** `drunn_row06_jump_attack.png` — 256×64 pixels.

---

## Prompt 8: MAGIC — Row 7 (8 frames, one-shot)

Generate a **horizontal sprite strip** of **8 frames**, each **64×64 pixels**, total **512×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Earth/fire magic — Drunn channels dwarven runic magic through his axes. Primal, rage-fueled. (Game logic handles VFX for three tiers.)
- Frame 0: Stance — plants feet wide, axes crossed before him.
- Frame 1–2: Channeling — raises both axes skyward. Runes on breastplate begin glowing. Beard stiffens/bristles with energy.
- Frame 3: Charging — trembling with power, ground cracking beneath (implied by pose). Eyes and rune glow intensely.
- Frame 4–5: Release — slams both axes into the ground, channeling energy outward. Massive impact pose.
- Frame 6: Shockwave — energy pulsing. Beard and braids blown outward.
- Frame 7: Recovery — pulling axes from ground, heavy breathing.
- Feet at y=56. 1px dark outline.

**Frame timing:** 140ms per frame (slowest magic — raw power).

**Output:** `drunn_row07_magic.png` — 512×64 pixels.

---

## Prompt 9: HIT — Row 8 (3 frames, one-shot)

Generate a **horizontal sprite strip** of **3 frames**, each **64×64 pixels**, total **192×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Barely flinching — Drunn has the most HP (10/10) and is the toughest. His hit reaction is small but noticeable.
- Frame 0: Impact — head tilts slightly, body shifts back barely. More annoyed than hurt.
- Frame 1: Grunt — eyes narrow, jaw clenches. Barely moves. Axes stay gripped.
- Frame 2: Recovering — already looking angry and ready to retaliate.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `drunn_row08_hit.png` — 192×64 pixels.

---

## Prompt 10: KNOCKDOWN — Row 9 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Toppled like a felled tree — the only thing that can actually knock Drunn down takes serious force.
- Frame 0: Impact — body jolts, helm rattled.
- Frame 1: Tipping — the wide, heavy body losing balance, axes flying outward.
- Frame 2: Falling — crashing sideways, beard and braids whipping.
- Frame 3: Fully prone — on his back, legs slightly raised, enormous mass flat on the ground. Helm askew.
- 1px dark outline.

**Frame timing:** 110ms per frame.

**Output:** `drunn_row09_knockdown.png` — 256×64 pixels.

---

## Prompt 11: GETUP — Row 10 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Slow, angry getup — pure stubborn dwarven will. No acrobatics here.
- Frame 0: Rolling over — grunting, pushing with arms.
- Frame 1: Pushing up — arms strain under his own weight.
- Frame 2: On one knee — scooping up axes, adjusting helm.
- Frame 3: Standing ready — full height again, axes in hand, absolutely furious.
- Feet at y=56 in final frame. 1px dark outline.

**Frame timing:** 130ms per frame (slowest getup — heaviest character).

**Output:** `drunn_row10_getup.png` — 256×64 pixels.

---

## Prompt 12: DEATH — Row 11 (6 frames, one-shot — holds on last frame)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Epic death — the mighty dwarf finally falls. Non-looping, holds on last frame.
- Frame 0: Fatal blow — head snaps, helm flies off.
- Frame 1: Staggering — dropping one axe, clutching wound.
- Frame 2: Knees buckling — falling forward.
- Frame 3: Crashing — massive body hitting the ground. Impact.
- Frame 4: Settling — dust implied, axes nearby, helm rolled away.
- Frame 5: **Final hold frame** — motionless on the ground. Face down. Beard spread. Slightly faded/darkened.
- 1px dark outline.

**Frame timing:** 140ms per frame.

**Output:** `drunn_row11_death.png` — 384×64 pixels.

---

## Prompt 13: MOUNT_IDLE — Row 12 (4 frames, looping)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. **Upper body only (waist up).**

**Animation:** Seated on a mount in idle. Only upper body rendered — mount is a separate sprite beneath.
- Both axes resting on thighs/lap. Heavy breathing. Beard sway.
- Character hips at y=48. Seamless loop.
- 1px dark outline.

**Frame timing:** 160ms per frame.

**Output:** `drunn_row12_mount_idle.png` — 256×64 pixels.

---

## Prompt 14: MOUNT_ATTACK — Row 13 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT. **Upper body only (waist up).**

**Animation:** Mounted axe swing — wide horizontal sweep while riding.
- Frame 0: Right axe reared back, left axe bracing.
- Frame 1: Twisting torso for momentum.
- Frame 2: **Active hit frame** — right axe sweeping in massive horizontal arc.
- Frame 3: Follow-through — body rotated.
- Frame 4: Recovery — resettling on mount.
- Character hips at y=48. 1px dark outline.

**Frame timing:** 90ms per frame.

**Output:** `drunn_row13_mount_attack.png` — 320×64 pixels.

---

## Prompt 15: RUN — Row 14 (6 frames, looping)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Lumbering run — Drunn is the slowest character (Speed 4/10). This is a determined, heavy charge, not a sprint.
- Short legs pumping, body low. More of a bull-charge than a run.
- Both axes held at sides, swinging with stride. Beard streaming behind. Helm bouncing slightly.
- Body barely lifts off ground — pure forward momentum.
- Seamless loop. Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame (slowest run of all characters).

**Output:** `drunn_row14_run.png` — 384×64 pixels.

---

## Prompt 16: THROW — Row 15 (6 frames, one-shot)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Drunn Ironhelm. Faces RIGHT.

**Animation:** Power throw — pure brute-force overhead hurl. Enemy is NOT shown. Drunn literally picks them up and tosses them like a sack of grain.
- Frame 0: Grab — axes stowed on belt, both hands grabbing forward.
- Frame 1: Lift — hoisting something heavy overhead with pure strength.
- Frame 2: Coiling — body twisting, loading the throw.
- Frame 3: Hurl — massive overhead toss forward. Arms fully extended.
- Frame 4: **Release frame** — enemy launched, follow-through.
- Frame 5: Recovery — retrieving axes, settling back.
- Feet at y=56. 1px dark outline.

**Frame timing:** 110ms per frame.

**Output:** `drunn_row15_throw.png` — 384×64 pixels.

---

## Final Assembly

After generating all 16 row strips, assemble them vertically in order into a single spritesheet:

```
Row  0: drunn_row00_idle.png         (384×64  → pad right to 896×64)
Row  1: drunn_row01_walk.png         (512×64  → pad right to 896×64)
Row  2: drunn_row02_attack1.png      (320×64  → pad right to 896×64)
Row  3: drunn_row03_attack2.png      (320×64  → pad right to 896×64)
Row  4: drunn_row04_attack3.png      (448×64  → pad right to 896×64)
Row  5: drunn_row05_jump.png         (256×64  → pad right to 896×64)
Row  6: drunn_row06_jump_attack.png  (256×64  → pad right to 896×64)
Row  7: drunn_row07_magic.png        (512×64  → pad right to 896×64)
Row  8: drunn_row08_hit.png          (192×64  → pad right to 896×64)
Row  9: drunn_row09_knockdown.png    (256×64  → pad right to 896×64)
Row 10: drunn_row10_getup.png        (256×64  → pad right to 896×64)
Row 11: drunn_row11_death.png        (384×64  → pad right to 896×64)
Row 12: drunn_row12_mount_idle.png   (256×64  → pad right to 896×64)
Row 13: drunn_row13_mount_attack.png (320×64  → pad right to 896×64)
Row 14: drunn_row14_run.png          (384×64  → pad right to 896×64)
Row 15: drunn_row15_throw.png        (384×64  → pad right to 896×64)
```

**Final spritesheet:** `drunn_spritesheet.png` — **896×1024 pixels** (14 columns × 64px, 16 rows × 64px).
Pad each row with transparent pixels on the right to reach 896px width.

**Place in:** `assets/sprites/drunn_spritesheet.png`
