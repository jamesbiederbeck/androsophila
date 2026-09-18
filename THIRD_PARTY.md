# DOOMFLY attribution and license scope

DOOMFLY is an independent research experiment, not affiliated with or endorsed by
id Software, Bethesda, ZeniMax, Freedoom, Farama Foundation, the dataset creators
or the cited researchers. DOOM and related marks belong to their respective
owners. This statement grants no trademark rights and is not trademark clearance.

## Original project material

The [MIT license](LICENSE) covers the original DOOMFLY Python/C++ code, original
application code, tests, documentation and custom arena definitions. Original
project contributions to generated app graphics are offered under MIT to the
extent the contributors hold rights in them; no exclusive copyright in purely
AI-generated elements is asserted. Existing third-party portions retain their
own terms. MIT does not relicense research data, game artwork or trademarks.

## MaleCNS v1.0 connectome, dopamine/memory source data and the LIF framework

The connectome engine this repository runs on — the native/GPU LIF kernel,
the MaleCNS importer, and the dopamine-gated-plasticity physiology code
(calibrated against Huang et al. 2024) — now lives in the separate
`connectome_sim/` submodule repository, along with the full attribution for
the MaleCNS v1.0 release (CC BY 4.0), the Huang et al. 2024 paper (CC BY 4.0)
and the Shiu et al. 2024 LIF framework (MIT). See `connectome_sim/THIRD_PARTY.md`.
This repository consumes that data and code via the submodule; it does not
bundle a separate copy.

## ViZDoom, Freedoom and the arena

The runtime pins ViZDoom 1.3.0 and explicitly selects its installed
`freedoom2.wad`. No commercial Doom campaign IWAD is distributed. The repository's
small WADs contain original map geometry, scripts and/or solid-color experimental
textures. Texture and actor names refer to the separately installed game assets.
Game screenshots and game-derived elements in app/social graphics contain
Freedoom artwork and retain its BSD-3-Clause attribution and disclaimer.

The full [Freedoom and ViZDoom notices](THIRD_PARTY_NOTICES.md) distinguish original
ViZDoom MIT code from the underlying engine's additional licensing. No engine
executable or upstream engine source checkout is bundled. A future redistributed
engine or container must retain all applicable upstream notices and obligations.

The optional arena compiler is [ZDoom ACC](https://github.com/ZDoom/acc/tree/bdb9bc4d2c5aee7ca3ff8da985d73aaf83af0557).
It is not bundled. To rebuild the WAD, clone that revision into `tools/acc/`,
follow its build instructions, and run `python -m doom.combat_arena --acc PATH_TO_ACC`.
Preserve the compiler's source notices if redistributing it. It is not required
at simulation runtime.

## UI and scientific model references

Adapted shadcn/ui components and the viewer's direct package dependencies retain
their full notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
The Silkscreen font is requested from Google Fonts, not bundled in this repository.
Dependency files themselves are installed from the lockfile.

Other scientific papers are cited in the methods and review documents; citing a
paper or implementing an equation does not grant rights to republish its figures.
