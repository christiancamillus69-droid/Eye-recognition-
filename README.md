# Rot & Bloom

An incremental game about a fungal network growing under a forest floor. You start by
digesting leaf litter by hand; you end up running something the size of a continent.

Open `index.html` in a browser. No build step, no dependencies, no network calls.
Progress saves automatically and keeps accruing while the tab is closed.

## The loop

**Digest** the substrate for biomass, spend biomass on **ten kinds of network** — from a
single Hypha up to a Continental Mat — and each one produces biomass on its own. Every
10, 25, 50, 100, 150 and 200 of a kind doubles that kind's yield, so counts matter as much
as tiers.

**Refinements** are one-time multipliers bought with biomass. They unlock as you qualify
for them and are lost on every bloom.

**Sporulation** is the prestige layer. Past 500K biomass in a single bloom you can collapse
the network: biomass, generators and refinements all reset, but you keep spores — and every
lifetime spore permanently raises all yields. Spores also buy **adaptations**, which survive
every future bloom: better offline rates, an auto-digester, cheaper generators, flat
production multipliers.

**Observations** are 18 achievements, each worth a further +1% to everything.

## Pacing

Measured by simulating a full playthrough rather than by feel:

| | reached at |
| --- | --- |
| Rhizomorph | 1m44s |
| Fruiting Body | 7m31s |
| Spore Print | 15m |
| First sporulation available | **17m41s** |
| Mycorrhizal Bridge | 26m |
| Fairy Ring | 43m |
| 1B biomass | 64m |
| Old-Growth Nexus | 133m |

Deep Vein and Continental Mat are still out of reach after three hours of a single
un-prestiged run, so the long tail stays intact. The first prestige lands inside twenty
minutes deliberately — the second layer is the most interesting part of the game and
shouldn't be hidden behind an hour of grinding.

## Notes

Single file. Canvas 2D draws the mycelium network, which grows as a function of biomass
earned this bloom — so it visibly regrows after each sporulation, over a faint ghost of the
largest network you have ever reached. `localStorage` for saves; time away pays out at 50%
(80% with Persistence), capped at 8 hours (24 with Deep Dormancy). Honors
`prefers-reduced-motion`.
