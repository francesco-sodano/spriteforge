# Spritesheet Generation Instructions — Sylara Windarrow (Ranger)

> **Workflow:** Generate the **Base Character Image** first to establish the design. Then generate each animation row **one at a time** as a separate horizontal sprite strip. Finally, assemble all 16 rows into the final 896×1024 spritesheet.

---

## Global Technical Rules (apply to ALL prompts below)

- **Art style:** Modern HD pixel art (detailed, clean pixel art at 2× density — like Dead Cells or Owlboy). NOT low-res NES-style.
- **Frame size:** Each frame is exactly **64×64 pixels**.
- **Background:** Fully **transparent** (alpha channel). Output as **PNG-32**.
- **No anti-aliasing to background** — crisp pixel-art edges against transparency.
- **Consistent 1-pixel dark outline** (dark teal or black) around the character in every frame.
- **Facing direction:** Character faces **RIGHT** in all frames.
- **Feet anchor:** Feet at approximately **y=56** within each 64×64 frame (8px from bottom) for ground alignment.
- **Character centered** horizontally within each 64×64 cell. Weapon swings and arrows may extend near edges but must not be clipped.
- **Unused cells** in a row must be fully transparent/empty.
- **No border/padding** between frames — tightly packed in the 64×64 grid.

---

## Prompt 0: Base Character Image

> **Purpose:** Generate a single **character reference sheet** to lock down the visual design before creating animation frames. This is NOT a sprite — it's a high-detail reference at a larger size.

**Prompt:**

Generate a pixel art character reference sheet for a fantasy elven ranger named **Sylara Windarrow**. Modern HD pixel art style (like Dead Cells / Owlboy). The sheet must follow the exact layout described below.

**Reference Sheet Layout — 1024×768 pixel canvas:**

The image must follow this exact layout so all three character sheets are visually consistent:

- **Background:** Solid blue (R=0, G=0, B=255) across the entire canvas. No gradients, no patterns.
- **Top banner (y: 0–64):**
  - Character name **"SYLARA WINDARROW"** in large **white (#FFFFFF) pixel-art uppercase text**, centered horizontally at y≈20.
  - Class label **"RANGER"** in smaller white text, centered below the name at y≈46.
- **Left panel — FRONT VIEW (x: 40–490, y: 80–620):**
  - Full-body neutral standing pose facing the viewer (front-facing), centered within the panel.
  - Character drawn at approximately **4× sprite scale** (~184–200px tall) for detail visibility.
  - Bow in left hand, quiver visible on back, pointed ears visible.
  - White label text **"FRONT"** centered below the character at y≈630.
- **Right panel — 3/4 SIDE VIEW (x: 534–984, y: 80–620):**
  - Same full-body neutral standing pose, rotated to a **3/4 right-facing perspective** (matching the in-game sprite facing direction).
  - Same 4× scale as the front view. Bow, quiver, flowing hair, and pointed ears clearly visible from this angle.
  - White label text **"3/4 SIDE"** centered below the character at y≈630.
- **Bottom strip — COLOR PALETTE (y: 660–740):**
  - A horizontal row of color swatches, evenly spaced and centered across the canvas width.
  - Each swatch is a **32×32 pixel filled square** with a **1px white border**.
  - Below each swatch: the element name in small white uppercase text (e.g., "SKIN", "HAIR", "VEST").
  - Show **P1 palette only** in swatch order matching the P1 table below (left-to-right: Skin, Hair, Eyes, Vest, Pants, Bracers, Boots, Bow, Blades, Fletching, Quiver).

**Character description:**
- **Build:** Slender, athletic, graceful — elven proportions. Taller than average human but lean. ~46–50 pixels tall in-game sprite. Slightly shorter and more slender than the warrior character (Theron).
- **Skin:** Light/fair with a slight warm tone (elven complexion) — RGB approximately (235, 210, 185).
- **Hair:** Long, golden blonde (220, 185, 90), straight, reaching mid-back. Flows freely — significant hair movement expected in action animations. Pointed elven ears visible through the hair.
- **Face:** Sharp elven features, high cheekbones, intense focused eyes — bright teal-green (50, 180, 140). Determined but elegant expression.
- **Ears:** Classic pointed elven ears, extending 3–4 pixels past the head silhouette.
- **Armor/Outfit — light ranger attire optimized for speed:**
  - Fitted forest-green (50, 100, 45) leather vest/jerkin with leaf-pattern embossing. No metal plates.
  - Small leaf-shaped leather shoulder pads, minimal profile.
  - Bare upper arms showing lean muscle, forearm-length leather archer's bracers (110, 75, 40) — right arm has a longer bracer for bowstring protection.
  - Dark green (40, 75, 35) fitted leather leggings/pants.
  - Lightweight dark brown (65, 45, 30) leather boots to mid-calf.
  - Leather utility belt at waist with small pouches. Dual-blade scabbard on left hip (cross-draw).
  - Leather quiver (110, 75, 40) visible on her back — arrow fletching (255, 255, 240) visible over right shoulder.
  - **No cape/cloak** — clean, unobstructed silhouette (contrasts with Theron's cloak).
- **Weapons:**
  - **Bow (primary):** Elegant elven recurve bow in left hand. Light tan/golden wood (180, 150, 90) with greenish vine wrapping. ~30 pixels tall (~2/3 of her height). String visible when drawn.
  - **Dual Blades (melee combo):** Two short curved leaf-shaped daggers. Silver steel blades (190, 200, 210) with green leather grips. Sheathed in belt scabbard when not in use. ~16 pixels each.
- **Distinctive silhouette features:** Quiver on back, pointed ears, long flowing hair, slender build, bow in left hand, no cape.

**P1 Color Palette:**

| Element | RGB |
|---|---|
| Skin | (235, 210, 185) |
| Hair | (220, 185, 90) |
| Eyes | (50, 180, 140) |
| Leather vest | (50, 100, 45) |
| Leather pants | (40, 75, 35) |
| Bracers/belt | (110, 75, 40) |
| Boots | (65, 45, 30) |
| Bow wood | (180, 150, 90) |
| Blade steel | (190, 200, 210) |
| Arrow fletching | (255, 255, 240) |
| Quiver | (110, 75, 40) |

**P2 Palette (alternate — for palette swap):**

| Element | P1 → P2 |
|---|---|
| Leather vest | (50, 100, 45) → (105, 105, 110) |
| Leather pants | (40, 75, 35) → (80, 80, 85) |
| Hair | (220, 185, 90) → (195, 200, 210) |

P2 must be **exact flat color swaps** — no gradients. The game engine does pixel-level color replacement.

**Output:** `sylara_base_reference.png` — **1024×768 pixels**. PNG with solid blue background (0, 0, 255).

---

## Prompt 1: IDLE — Row 0 (6 frames, looping)

Generate a **horizontal sprite strip** of **6 frames**, each frame **64×64 pixels**, for a total image size of **384×64 pixels**. PNG-32 with transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow — elven ranger with forest-green leather vest, long golden blonde hair, pointed ears, elven recurve bow in left hand, quiver on back. Faces RIGHT. (Use the base character reference for visual consistency.)

**Animation:** Relaxed ready stance.
- Sylara stands with bow held loosely in left hand at her side, weight shifted to one hip.
- Hair sways gently. Quiver visible on back. Ears visible. Elegantly poised.
- Frame 0–5: Subtle breathing — chest rise/fall. Occasional head turn or hair flick. Loops seamlessly (frame 5 → frame 0).
- Feet at y=56. 1px dark outline. No anti-aliasing to background.

**Frame timing:** 150ms per frame.

**Output:** `sylara_row00_idle.png` — 384×64 pixels.

---

## Prompt 2: WALK — Row 1 (8 frames, looping)

Generate a **horizontal sprite strip** of **8 frames**, each **64×64 pixels**, total **512×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Graceful, light-footed walk cycle — elven agility, almost gliding.
- Bow held down in left hand. Hair flows behind her with movement.
- Barely any vertical bobbing — smooth, cat-like stride.
- Feet cycle: contact → pass → contact → pass (8-frame walk). Seamless loop.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `sylara_row01_walk.png` — 512×64 pixels.

---

## Prompt 3: ATTACK1 — Row 2 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Quick dual-blade slash — right blade sweeps right to left (forehand). First hit of a 3-hit melee combo. She switches from bow to dual daggers for melee attacks.
- Frame 0: Drawing blades from belt — left hand drops bow (bow disappears, considered stowed).
- Frame 1: Blade in right hand cocked back.
- Frame 2: **Active hit frame** — right blade fully extended in a horizontal arc. Reach ~16px past body.
- Frame 3: Slash follow-through.
- Frame 4: Recovery — left blade ready for next attack.
- Feet at y=56. 1px dark outline.

**Frame timing:** 70ms per frame (fast — Speed 10/10).

**Output:** `sylara_row02_attack1.png` — 320×64 pixels.

---

## Prompt 4: ATTACK2 — Row 3 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT. Wielding dual curved daggers (bow stowed).

**Animation:** Left blade backhand slash — opposite direction from ATTACK1. Second hit of combo — flows continuously like a dance.
- Flowing combo from ATTACK1 — continuous, dance-like motion.
- Frame 2: **Active hit frame** — left blade fully extended.
- Body spins slightly — rotates during the dual-blade choreography.
- Feet at y=56. 1px dark outline.

**Frame timing:** 65ms per frame.

**Output:** `sylara_row03_attack2.png` — 320×64 pixels.

---

## Prompt 5: ATTACK3 — Row 4 (7 frames, one-shot)

Generate a **horizontal sprite strip** of **7 frames**, each **64×64 pixels**, total **448×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT. Wielding dual curved daggers.

**Animation:** Spinning dual-blade cross-slash — both blades sweep in an X pattern. Combo finisher, causes enemy knockdown.
- Frame 0–1: Wind-up — body crouches, both blades crossed at chest.
- Frame 2: Spinning — body rotates, hair whipping around.
- Frame 3: **Active hit frame** — both blades extended outward in a cross-slash, maximum reach.
- Frame 4: Impact — blades at full extension.
- Frame 5–6: Recovery — landing from spin, returning to stance. Blades re-sheathed, bow returns to hand.
- Feet at y=56. 1px dark outline.

**Frame timing:** 80ms per frame.

**Output:** `sylara_row04_attack3.png` — 448×64 pixels.

---

## Prompt 6: JUMP — Row 5 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT. Bow in left hand.

**Animation:** Acrobatic jump — more graceful than other characters, showing elven agility.
- Frame 0: Crouch — low squat, coiled to spring.
- Frame 1: Launch — body spinning/flipping upward, hair trailing behind.
- Frame 2: Apex — airborne, legs tucked, hair floating. Bow held to side.
- Frame 3: Descent — legs extending for landing, hair floating above.
- Feet at y=56 in frames 0 and 3. 1px dark outline.

**Frame timing:** 90ms per frame.

**Output:** `sylara_row05_jump.png` — 256×64 pixels.

---

## Prompt 7: JUMP_ATTACK — Row 6 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT. Bow in hand.

**Animation:** Aerial downward bow shot — fires an arrow straight down while airborne.
- Frame 0: Airborne — drawing arrow from quiver.
- Frame 1: Nocking arrow — bow aimed downward at angle.
- Frame 2: **Active hit frame** — bowstring released, arrow leaving bow downward. Body angled down.
- Frame 3: Landing — bow returning to rest, landing crouch.
- 1px dark outline.

**Frame timing:** 80ms per frame.

**Output:** `sylara_row06_jump_attack.png` — 256×64 pixels.

---

## Prompt 8: MAGIC — Row 7 (8 frames, one-shot)

Generate a **horizontal sprite strip** of **8 frames**, each **64×64 pixels**, total **512×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Channeling nature/storm magic. Dramatic spell-casting sequence (game logic handles different VFX overlays for three tiers).
- Frame 0: Stance — plants feet apart, bow held horizontally.
- Frame 1–2: Drawing power — raises free hand skyward, hair begins floating upward with magical energy. Green/teal glow forms around her.
- Frame 3: Charging — both arms raised, nature energy spiraling around her. Hair fully floating.
- Frame 4–5: Release — arms thrust forward/outward, energy bursts forth. Dramatic pose with hair streaming.
- Frame 6: Power dissipating — lowering arms, wind dying down.
- Frame 7: Recovery — settling back to neutral, hair falling back to rest.
- Feet at y=56. 1px dark outline.

**Frame timing:** 120ms per frame.

**Output:** `sylara_row07_magic.png` — 512×64 pixels.

---

## Prompt 9: HIT — Row 8 (3 frames, one-shot)

Generate a **horizontal sprite strip** of **3 frames**, each **64×64 pixels**, total **192×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Recoil from being struck — more dramatic than Theron since she has less HP (lighter build, bigger reaction).
- Frame 0: Impact — head whips back, hair flies forward, body jerks backward.
- Frame 1: Stagger — bent at waist, one arm clutching where hit.
- Frame 2: Recovering — straightening up, hair settling.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `sylara_row08_hit.png` — 192×64 pixels.

---

## Prompt 10: KNOCKDOWN — Row 9 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Knocked to the ground.
- Frame 0: Impact — body jolts.
- Frame 1: Falling — body arching backward, hair whipping.
- Frame 2: Hitting ground — on her side.
- Frame 3: Fully prone — lying still, hair spread on ground.
- 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `sylara_row09_knockdown.png` — 256×64 pixels.

---

## Prompt 11: GETUP — Row 10 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Recovering from knockdown — acrobatic recovery fitting her agile nature.
- Frame 0: Rolling — pushing up from ground.
- Frame 1: Kip-up — springing up with momentum.
- Frame 2: Landing on feet — slightly crouched.
- Frame 3: Ready stance — bow in hand, combat-ready.
- Feet at y=56 in final frame. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `sylara_row10_getup.png` — 256×64 pixels.

---

## Prompt 12: DEATH — Row 11 (6 frames, one-shot — holds on last frame)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Final death sequence. Non-looping — the game holds on the last frame. More dramatic, with significant hair movement.
- Frame 0: Fatal hit — extreme recoil, bow falling from hand.
- Frame 1: Stumbling — staggering.
- Frame 2: Knees giving way — collapsing gracefully.
- Frame 3: Falling — body collapsing to side.
- Frame 4: On ground — lying on side, bow nearby, quiver spilled.
- Frame 5: **Final hold frame** — still, motionless. Hair spread around head. Slightly faded/darkened.
- 1px dark outline.

**Frame timing:** 130ms per frame.

**Output:** `sylara_row11_death.png` — 384×64 pixels.

---

## Prompt 13: MOUNT_IDLE — Row 12 (4 frames, looping)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT. **Upper body only (waist up).**

**Animation:** Seated on a mount in idle state. Only the upper body is rendered — the mount is a separate sprite beneath.
- Bow held upright in left hand, looking forward. Hair sways gently. Alert, scanning.
- Character hips at y=48 (seated position). Seamless loop.
- 1px dark outline.

**Frame timing:** 150ms per frame.

**Output:** `sylara_row12_mount_idle.png` — 256×64 pixels.

---

## Prompt 14: MOUNT_ATTACK — Row 13 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT. **Upper body only (waist up).**

**Animation:** Mounted bow shot — fires arrow horizontally while riding.
- Frame 0: Drawing arrow from quiver.
- Frame 1: Nocking and aiming — bow drawn, aiming right.
- Frame 2: **Active hit frame** — bowstring released, arrow launching to the right.
- Frame 3: Follow-through — bow arm extended.
- Frame 4: Recovery — lowering bow.
- Character hips at y=48. 1px dark outline.

**Frame timing:** 80ms per frame.

**Output:** `sylara_row13_mount_attack.png` — 320×64 pixels.

---

## Prompt 15: RUN — Row 14 (6 frames, looping)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Fast sprint — Sylara is the fastest character (Speed 10/10). Very dynamic, energetic motion.
- Full sprint, body leaned forward at sharper angle than walking.
- Hair streaming far behind her.
- Bow held in left hand at side. Legs show long, rapid strides.
- Seamless loop. Feet at y=56. 1px dark outline.

**Frame timing:** 70ms per frame (fastest run of all characters).

**Output:** `sylara_row14_run.png` — 384×64 pixels.

---

## Prompt 16: THROW — Row 15 (6 frames, one-shot)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Sylara Windarrow. Faces RIGHT.

**Animation:** Grab-and-throw — technique-based, less brute-force than other characters. Enemy is NOT shown.
- Frame 0: Grab — reaching with right hand, bow stowed.
- Frame 1–2: Leverage — using opponent's momentum, body pivoting.
- Frame 3: Flip — executing a hip throw / judo-style toss.
- Frame 4: **Release frame** — enemy launched, arms follow through.
- Frame 5: Recovery — returning to stance, bow back in hand.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `sylara_row15_throw.png` — 384×64 pixels.

---

## Final Assembly

After generating all 16 row strips, assemble them vertically in order into a single spritesheet:

```
Row  0: sylara_row00_idle.png         (384×64  → pad right to 896×64)
Row  1: sylara_row01_walk.png         (512×64  → pad right to 896×64)
Row  2: sylara_row02_attack1.png      (320×64  → pad right to 896×64)
Row  3: sylara_row03_attack2.png      (320×64  → pad right to 896×64)
Row  4: sylara_row04_attack3.png      (448×64  → pad right to 896×64)
Row  5: sylara_row05_jump.png         (256×64  → pad right to 896×64)
Row  6: sylara_row06_jump_attack.png  (256×64  → pad right to 896×64)
Row  7: sylara_row07_magic.png        (512×64  → pad right to 896×64)
Row  8: sylara_row08_hit.png          (192×64  → pad right to 896×64)
Row  9: sylara_row09_knockdown.png    (256×64  → pad right to 896×64)
Row 10: sylara_row10_getup.png        (256×64  → pad right to 896×64)
Row 11: sylara_row11_death.png        (384×64  → pad right to 896×64)
Row 12: sylara_row12_mount_idle.png   (256×64  → pad right to 896×64)
Row 13: sylara_row13_mount_attack.png (320×64  → pad right to 896×64)
Row 14: sylara_row14_run.png          (384×64  → pad right to 896×64)
Row 15: sylara_row15_throw.png        (384×64  → pad right to 896×64)
```

**Final spritesheet:** `sylara_spritesheet.png` — **896×1024 pixels** (14 columns × 64px, 16 rows × 64px).
Pad each row with transparent pixels on the right to reach 896px width.

**Place in:** `assets/sprites/sylara_spritesheet.png`
