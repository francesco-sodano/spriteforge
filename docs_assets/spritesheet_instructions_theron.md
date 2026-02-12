# Spritesheet Generation Instructions — Theron Ashblade (Warrior)

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
- **Character centered** horizontally within each 64×64 cell. Weapon swings may extend near edges but must not be clipped.
- **Unused cells** in a row must be fully transparent/empty.
- **No border/padding** between frames — tightly packed in the 64×64 grid.

---

## Prompt 0: Base Character Image

> **Purpose:** Generate a single **character reference sheet** to lock down the visual design before creating animation frames. This is NOT a sprite — it's a high-detail reference at a larger size.

**Prompt:**

Generate a pixel art character reference sheet for a fantasy warrior named **Theron Ashblade**. Modern HD pixel art style (like Dead Cells / Owlboy). The sheet must follow the exact layout described below.

**Reference Sheet Layout — 1024×768 pixel canvas:**

The image must follow this exact layout so all three character sheets are visually consistent:

- **Background:** Solid blue (R=0, G=0, B=255) across the entire canvas. No gradients, no patterns.
- **Top banner (y: 0–64):**
  - Character name **"THERON ASHBLADE"** in large **white (#FFFFFF) pixel-art uppercase text**, centered horizontally at y≈20.
  - Class label **"WARRIOR"** in smaller white text, centered below the name at y≈46.
- **Left panel — FRONT VIEW (x: 40–490, y: 80–620):**
  - Full-body neutral standing pose facing the viewer (front-facing), centered within the panel.
  - Character drawn at approximately **4× sprite scale** (~192–208px tall) for detail visibility.
  - Weapon (Emberfang longsword) in right hand, crimson cape visible behind.
  - White label text **"FRONT"** centered below the character at y≈630.
- **Right panel — 3/4 SIDE VIEW (x: 534–984, y: 80–620):**
  - Same full-body neutral standing pose, rotated to a **3/4 right-facing perspective** (matching the in-game sprite facing direction).
  - Same 4× scale as the front view. Sword, cape, and asymmetric pauldrons clearly visible from this angle.
  - White label text **"3/4 SIDE"** centered below the character at y≈630.
- **Bottom strip — COLOR PALETTE (y: 660–740):**
  - A horizontal row of color swatches, evenly spaced and centered across the canvas width.
  - Each swatch is a **32×32 pixel filled square** with a **1px white border**.
  - Below each swatch: the element name in small white uppercase text (e.g., "SKIN", "HAIR", "ARMOR").
  - Show **P1 palette only** in swatch order matching the P1 table below (left-to-right: Skin, Hair, Breastplate, Tunic/Cloak, Leather, Steel trim, Blade, Boots).

**Character description:**
- **Build:** Tall, muscular, athletic — classic heroic warrior proportions. ~48–52 pixels tall and ~24–28 pixels wide in-game sprite.
- **Skin:** Fair/tanned Caucasian skin — RGB approximately (210, 170, 130).
- **Hair:** Dark brown (60, 40, 25), medium-length, swept back or tied. Slight movement expected in animations.
- **Face:** Strong jaw, determined expression, short stubble or clean-shaven.
- **Armor — layered medieval plate-and-leather:**
  - Dark steel breastplate (80, 80, 95) over a crimson/dark red tunic (150, 30, 30).
  - Asymmetric steel pauldrons — larger on left/shield side.
  - Leather bracers with steel plates on forearms, bare forearms showing muscle.
  - Dark leather pants (100, 65, 35) with steel knee guards, heavy dark brown boots (50, 35, 25).
  - Short tattered crimson cloak (150, 30, 30) flowing from left shoulder — reacts to movement.
- **Weapon — Emberfang:** A flame-forged longsword held in right hand.
  - Slightly curved, single-edge bastard sword style.
  - Dark steel blade (140, 140, 150) with a faint orange/ember glow (255, 140, 40) along the cutting edge — subtle warm tinge, not full fire.
  - Bronze cross-guard (200, 190, 160), dark leather grip.
  - ~28–32 pixels long in sprite scale.
- **Distinctive silhouette features:** Cape/cloak from left shoulder, asymmetric pauldrons, longsword in right hand.

**P1 Color Palette:**

| Element | RGB |
|---|---|
| Skin | (210, 170, 130) |
| Hair | (60, 40, 25) |
| Breastplate | (80, 80, 95) |
| Tunic / Cloak | (150, 30, 30) |
| Leather | (100, 65, 35) |
| Steel trim | (200, 190, 160) |
| Emberfang blade | (140, 140, 150) + (255, 140, 40) edge glow |
| Boots | (50, 35, 25) |

**P2 Palette (alternate — for palette swap):**

| Element | P1 → P2 |
|---|---|
| Tunic / Cloak | (150, 30, 30) → (60, 60, 65) |
| Steel trim | (200, 190, 160) → (180, 185, 195) |
| Breastplate | (80, 80, 95) → (50, 55, 65) |

P2 must be **exact flat color swaps** — no gradients. The game engine does pixel-level color replacement.

**Output:** `theron_base_reference.png` — **1024×768 pixels**. PNG with solid blue background (0, 0, 255).

---

## Prompt 1: IDLE — Row 0 (6 frames, looping)

Generate a **horizontal sprite strip** of **6 frames**, each frame **64×64 pixels**, for a total image size of **384×64 pixels**. PNG-32 with transparent background. Modern HD pixel art.

**Character:** Theron Ashblade — warrior with dark steel breastplate, crimson tunic/cloak, longsword (Emberfang) in right hand. Faces RIGHT. (Use the base character reference for visual consistency.)

**Animation:** Subtle breathing / idle combat stance.
- Theron stands in a relaxed combat-ready stance, sword held in right hand pointing slightly downward and to the side. Weight shifted slightly to one leg.
- Frame 0–5: Gentle chest rise/fall, cloak swaying slightly, very subtle head movement. Loops seamlessly (frame 5 → frame 0).
- Feet anchored at y=56. 1px dark outline. No anti-aliasing to background.

**Frame timing:** 150ms per frame.

**Output:** `theron_row00_idle.png` — 384×64 pixels.

---

## Prompt 2: WALK — Row 1 (8 frames, looping)

Generate a **horizontal sprite strip** of **8 frames**, each **64×64 pixels**, total **512×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Standard walk cycle moving right.
- Purposeful stride. Sword held at side in right hand.
- Cape/cloak bounces with each step. Arms swing naturally (sword arm moves less).
- Feet cycle: contact → pass → contact → pass (standard 8-frame walk). Seamless loop.
- Feet at y=56. 1px dark outline. No anti-aliasing to background.

**Frame timing:** 100ms per frame.

**Output:** `theron_row01_walk.png` — 512×64 pixels.

---

## Prompt 3: ATTACK1 — Row 2 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Quick horizontal sword slash, right to left (forehand swing). First hit of a 3-hit combo.
- Frame 0: Wind-up — sword pulled back behind shoulder.
- Frame 1: Swing begins — sword arcs forward.
- Frame 2: **Active hit frame** — sword fully extended, maximum reach. Blade extends ~20px beyond body to the right.
- Frame 3: Follow-through — sword passes centerline.
- Frame 4: Recovery — returning toward neutral, anticipating ATTACK2.
- Feet at y=56. 1px dark outline.

**Frame timing:** 80ms per frame.

**Output:** `theron_row02_attack1.png` — 320×64 pixels.

---

## Prompt 4: ATTACK2 — Row 3 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Reverse backhand horizontal slash, left to right. Second hit of a 3-hit combo — flows naturally from ATTACK1's follow-through.
- Frame 0: Sword at end of ATTACK1's follow-through position (natural transition).
- Frame 2: **Active hit frame** — sword fully extended in opposite arc.
- Slightly faster feel than ATTACK1 (combo picks up speed).
- Feet at y=56. 1px dark outline.

**Frame timing:** 70ms per frame.

**Output:** `theron_row03_attack2.png` — 320×64 pixels.

---

## Prompt 5: ATTACK3 — Row 4 (7 frames, one-shot)

Generate a **horizontal sprite strip** of **7 frames**, each **64×64 pixels**, total **448×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Powerful overhead downward chop — the combo finisher that causes enemy knockdown.
- Frame 0–1: Wind-up — sword raised high above head with both hands.
- Frame 2: Sword starts downward arc.
- Frame 3: **Active hit frame** — sword swings down through center, maximum impact.
- Frame 4: Impact pose — deep lunge, sword near ground with sparks/impact emphasis.
- Frame 5–6: Recovery — slow return to standing. Cape dramatically settling.
- Feet at y=56. 1px dark outline.

**Frame timing:** 90ms per frame.

**Output:** `theron_row04_attack3.png` — 448×64 pixels.

---

## Prompt 6: JUMP — Row 5 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Vertical jump arc.
- Frame 0: Crouch/squat — knees bent, preparing to jump.
- Frame 1: Launch — legs extending, character rising, arms up.
- Frame 2: Apex — airborne, knees slightly tucked, sword held defensively.
- Frame 3: Landing — legs extending down for landing.
- Feet at y=56 in frames 0 and 3. Airborne frames centered vertically. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `theron_row05_jump.png` — 256×64 pixels.

---

## Prompt 7: JUMP_ATTACK — Row 6 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Aerial downward sword strike.
- Frame 0: Airborne — sword raised above head.
- Frame 1: Sword begins downward arc while falling.
- Frame 2: **Active hit frame** — full downward slash, body angled diagonally downward.
- Frame 3: Landing impact — deep crouch with sword planted forward.
- 1px dark outline. No anti-aliasing to background.

**Frame timing:** 80ms per frame.

**Output:** `theron_row06_jump_attack.png` — 256×64 pixels.

---

## Prompt 8: MAGIC — Row 7 (8 frames, one-shot)

Generate a **horizontal sprite strip** of **8 frames**, each **64×64 pixels**, total **512×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Channeling fire magic through Emberfang. Dramatic spell-casting sequence.
- Frame 0: Stance change — plants feet wide.
- Frame 1–2: Raises sword vertically overhead, both hands on hilt.
- Frame 3–4: Blade ignites — sword glows brighter orange/red (ember effect intensifies on the blade).
- Frame 5: Power release — sword thrust forward/upward, body leaning forward, fire burst emanating from blade.
- Frame 6–7: Settling — returning to neutral, brief exhausted pose.
- Feet at y=56. 1px dark outline.

**Frame timing:** 120ms per frame.

**Output:** `theron_row07_magic.png` — 512×64 pixels.

---

## Prompt 9: HIT — Row 8 (3 frames, one-shot)

Generate a **horizontal sprite strip** of **3 frames**, each **64×64 pixels**, total **192×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Recoil from being struck.
- Frame 0: Impact — head snaps back, body shifts backward, one foot slides back.
- Frame 1: Stagger — bent slightly at waist, pained expression.
- Frame 2: Recovering — straightening up, transitioning back to ready stance.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `theron_row08_hit.png` — 192×64 pixels.

---

## Prompt 10: KNOCKDOWN — Row 9 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Knocked to the ground by a heavy hit.
- Frame 0: Impact — body jolts backward.
- Frame 1: Falling — body tilting back, feet leaving ground.
- Frame 2: On ground — lying flat on back, sword dropped to side.
- Frame 3: Fully prone — flat on ground, still.
- 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `theron_row09_knockdown.png` — 256×64 pixels.

---

## Prompt 11: GETUP — Row 10 (4 frames, one-shot)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Recovering from knockdown to standing.
- Frame 0: Pushing up — one arm propping up body.
- Frame 1: Rising to one knee, reaching for sword.
- Frame 2: Standing up, sword retrieved.
- Frame 3: Ready stance — combat-ready (similar to IDLE frame 0).
- Feet at y=56 in final frame. 1px dark outline.

**Frame timing:** 120ms per frame.

**Output:** `theron_row10_getup.png` — 256×64 pixels.

---

## Prompt 12: DEATH — Row 11 (6 frames, one-shot — holds on last frame)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Final death sequence. Non-looping — the game holds on the last frame.
- Frame 0: Fatal hit — dramatic recoil, more extreme than HIT.
- Frame 1: Stumbling — staggering forward.
- Frame 2: Knees buckling — dropping to knees.
- Frame 3: Falling forward — body collapsing.
- Frame 4: On ground — lying face-down, sword falling from hand.
- Frame 5: **Final hold frame** — body flat, motionless. Slightly faded/darkened to indicate finality.
- 1px dark outline.

**Frame timing:** 130ms per frame.

**Output:** `theron_row11_death.png` — 384×64 pixels.

---

## Prompt 13: MOUNT_IDLE — Row 12 (4 frames, looping)

Generate a **horizontal sprite strip** of **4 frames**, each **64×64 pixels**, total **256×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT. **Upper body only (waist up).**

**Animation:** Seated on a mount in idle state. Only the upper body is rendered — the mount is a separate sprite beneath.
- Seated pose, sword held upright in right hand. Subtle breathing/sway. Seamless loop.
- Character hips aligned at y=48 within frame (lower than standing since legs are on mount).
- 1px dark outline.

**Frame timing:** 150ms per frame.

**Output:** `theron_row12_mount_idle.png` — 256×64 pixels.

---

## Prompt 14: MOUNT_ATTACK — Row 13 (5 frames, one-shot)

Generate a **horizontal sprite strip** of **5 frames**, each **64×64 pixels**, total **320×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT. **Upper body only (waist up).**

**Animation:** Mounted horizontal sword slash while riding.
- Wide sweeping slash to the right — sword extends far to cover mount's width.
- Frame 2: **Active hit frame** — sword fully extended at maximum reach.
- Character hips at y=48. 1px dark outline.

**Frame timing:** 80ms per frame.

**Output:** `theron_row13_mount_attack.png` — 320×64 pixels.

---

## Prompt 15: RUN — Row 14 (6 frames, looping)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Fast sprint animation.
- Full sprint — body leaning forward, cape streaming behind.
- Sword held at side, ready to strike. More dynamic arm/leg motion than WALK.
- Seamless loop. Feet at y=56. 1px dark outline.

**Frame timing:** 80ms per frame.

**Output:** `theron_row14_run.png` — 384×64 pixels.

---

## Prompt 16: THROW — Row 15 (6 frames, one-shot)

Generate a **horizontal sprite strip** of **6 frames**, each **64×64 pixels**, total **384×64 pixels**. PNG-32, transparent background. Modern HD pixel art.

**Character:** Theron Ashblade. Faces RIGHT.

**Animation:** Grab-and-throw animation. Enemy is NOT shown — only Theron's body posture implies the grapple.
- Frame 0: Grab — reaching forward with left hand, sword stowed.
- Frame 1–2: Lifting — hoisting enemy overhead with both hands.
- Frame 3: Wind-up — enemy overhead, body twisted.
- Frame 4: **Release frame** — throwing motion, body lunging forward.
- Frame 5: Follow-through — arms extended from throw, returning to stance.
- Feet at y=56. 1px dark outline.

**Frame timing:** 100ms per frame.

**Output:** `theron_row15_throw.png` — 384×64 pixels.

---

## Final Assembly

After generating all 16 row strips, assemble them vertically in order into a single spritesheet:

```
Row  0: theron_row00_idle.png         (384×64  → pad right to 896×64)
Row  1: theron_row01_walk.png         (512×64  → pad right to 896×64)
Row  2: theron_row02_attack1.png      (320×64  → pad right to 896×64)
Row  3: theron_row03_attack2.png      (320×64  → pad right to 896×64)
Row  4: theron_row04_attack3.png      (448×64  → pad right to 896×64)
Row  5: theron_row05_jump.png         (256×64  → pad right to 896×64)
Row  6: theron_row06_jump_attack.png  (256×64  → pad right to 896×64)
Row  7: theron_row07_magic.png        (512×64  → pad right to 896×64)
Row  8: theron_row08_hit.png          (192×64  → pad right to 896×64)
Row  9: theron_row09_knockdown.png    (256×64  → pad right to 896×64)
Row 10: theron_row10_getup.png        (256×64  → pad right to 896×64)
Row 11: theron_row11_death.png        (384×64  → pad right to 896×64)
Row 12: theron_row12_mount_idle.png   (256×64  → pad right to 896×64)
Row 13: theron_row13_mount_attack.png (320×64  → pad right to 896×64)
Row 14: theron_row14_run.png          (384×64  → pad right to 896×64)
Row 15: theron_row15_throw.png        (384×64  → pad right to 896×64)
```

**Final spritesheet:** `theron_spritesheet.png` — **896×1024 pixels** (14 columns × 64px, 16 rows × 64px).
Pad each row with transparent pixels on the right to reach 896px width.

**Place in:** `assets/sprites/theron_spritesheet.png`
