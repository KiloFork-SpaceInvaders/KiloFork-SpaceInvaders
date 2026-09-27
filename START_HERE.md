# Start Here — KiloFork Space Invaders

This repository is a downstream evolution of Neural Lab's compact Pygame Space Invaders project into a deterministic co-op shmup and AI-training testbed.

The original upstream remains the lineage/control specimen. This fork owns the 3044 evolution, not the upstream project's identity.

## Read first

1. `.gsv/project.yaml`
2. `README.md`
3. `AGENTS.md`
4. `docs/ARCHITECTURE_3044.md`
5. `docs/PLAYER_TWO_INTEGRATION.md`
6. `docs/ROADMAP_3044.md`

## Lineage boundary

Upstream:

```text
theneurallab/tiktok-space-invaders
```

Preserve its MIT attribution and history. Do not rewrite the repository as though 3044 originated the base game/assets.

The fork's local authority begins at the deterministic game-core/controller/training evolution documented here.

## Smallest verification

Renderer-free proof:

```sh
python -m unittest discover -s tests -v
python -m space_invaders.simulate --episodes 2 --seed 3044 --max-ticks 600
```

Visual/audio/input claims require a real Pygame run. A headless pass cannot prove them.

## Architecture rule

```text
controllers -> Action values -> deterministic core -> events/state
                                  |              |
                                  v              +-> telemetry/training
                               renderer
```

- gameplay rules belong in `space_invaders/core.py`;
- Pygame/input/audio belong in `space_invaders/app.py`;
- training contracts must remain renderer-free;
- Player Two integration enters through the controller/action boundary, never a sibling checkout import.

## Evidence ladder

Keep these separate:

```text
source compiles
!= pure core tests pass
!= headless deterministic episode
!= Pygame imports
!= window/input/audio verified
!= two-human co-op verified
!= AI wingman observed in real play
!= learned policy beats a named baseline
```

## Cross-lattice rule

Harvest contracts and lessons, not sibling source trees. If Kilo_Core, Player Two, Game Archaeology Lab, or another GSV surface needs this game, use the observation/action/episode contract or a versioned adapter.

Do not add `../SiblingRepo` path coupling.
