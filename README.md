# decam-chatterbox

Low-latency Slack reporting of gravitational-wave alerts for DECam.

This is a sibling of [chatterbox](https://github.com/lsst-so/chatterbox), which
does the same job for Rubin. It deliberately covers less ground: chatterbox
decodes a Rubin ToO producer's *derived* record and then launches a scheduler
simulation to say what Rubin will actually observe. decam-chatterbox decodes
the LIGO/Virgo/KAGRA (LVK) alert notice itself -- there is no producer in
front of it -- and stops after answering "what is this, and can DECam even
see it tonight". No observing strategy, no tiling plan, no simulation: that is
left for later, and for a human.

## What it posts

**One message per notice** (typically a few seconds, dominated by rendering
the dark-hours plot):

- **What the notice says**: alert type (`PRELIMINARY`/`INITIAL`/`UPDATE`/
  `RETRACTION`/`EARLYWARNING`), FAR and significance, CBC classification
  (BNS/NSBH/BBH/Terrestrial) and properties (HasNS/HasRemnant/HasMassGap/
  HasSSM), or duration and central frequency for a Burst event. A retraction
  gets a short message of its own: human vetting concluded the candidate is
  probably not astrophysical, and the notice carries nothing else.
- **The localization**, decoded from the multi-order sky map every notice
  (other than a retraction) carries inline: 90% area, centroid, declination
  and Galactic-latitude range, distance, and how many DECam pointings
  (3 deg^2 each) the area corresponds to -- stated only to describe the
  region's size, not as a tiling plan.
- **Sun and Moon statistics** for the relevant night at CTIO: sunset/sunrise,
  the -12 deg and -18 deg crossings, moonrise/moonset, illumination, and the
  Moon's separation from the localization centroid. In UTC and Chile local
  time.
- **Weather and site links** for Cerro Tololo.
- **Localization contour over accessible dark hours**: a full-sky map coloured
  by hours per pixel with airmass < 2 while the Sun is below -12 deg, with the
  localization drawn on top -- both the plain fraction of the region that is
  ever accessible and the *probability-weighted* mean accessible time are
  reported, because they can disagree: a region can be half accessible by area
  while almost none of its probability ever rises (a real, if unlucky,
  example is in `tests/data/samples` -- see below).
- **A chirp-mass line**, best-effort from GraceDB: the one number in this
  message the notice schema has no field for.

**And if decam-chatterbox itself fails, it says so in the channel.** An alert
nobody hears about is indistinguishable from no alert at all, so a decode
failure, a broken plot, a dead monitoring stream all get posted, naming what
was being attempted, the event id when known, and the exception. The failure
path is itself guarded, so a broken channel cannot turn one failure into two.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[gw,scimma,slack,test]"
```

Extras and what they buy:

| Extra    | Needed for                                                        |
| -------- | ------------------------------------------------------------------ |
| `gw`     | `astroplan` (CTIO almanac), `ligo.skymap` (reading the embedded skymap, the sky projections), `ligo-gracedb` (the optional chirp-mass lookup) |
| `scimma` | the SCIMMA ingest source (`hop-client`) and reading `.avro` records (`fastavro`) |
| `slack`  | actually posting (`slack-sdk`); without it, output is local         |
| `test`   | `pytest`, `ruff`, `black`, `isort`                                  |

Without `gw`, decam-chatterbox cannot process any notice at all -- unlike
chatterbox's Rubin almanac (an optional extra there because a Rubin site
already has it installed for other reasons), CTIO observability is this
tool's whole job.

## Configure

```bash
cp config.yaml.example config.yaml
```

Then set the bot token in the environment -- it is never stored in the config:

```bash
export SLACK_BOT_TOKEN=xoxb-...
```

The token needs the `chat:write` and `files:write` scopes. A bot token is
required rather than an incoming webhook because webhooks cannot upload
files, and every post carries the dark-hours plot.

## Where alerts come from

LVK alerts are distributed over Kafka in two equivalent ways: as JSON from
NASA's GCN, or as Avro from SCIMMA's Hopskotch broker. **decam-chatterbox
uses SCIMMA** (`ingest.kind: scimma`, the default), via `hop-client`:

1. Sign up for a SCIMMA account at <https://hop.scimma.org> (a GitHub or
   Google account works if your institution is not in its identity-provider
   list).
2. Create credentials at <https://my.hop.scimma.org/> and save the generated
   password -- it cannot be retrieved again later.
3. Click "Manage" next to that credential and subscribe to the `igwn.gwalert`
   topic. A fresh subscription can take up to about an hour to become usable.
4. `hop auth add` and enter that username and password. This writes
   `~/.config/hop/auth.toml`, which is the only place the credentials are
   kept -- `ingest.hop_url` in the config just names the broker and topic
   (`kafka://kafka.scimma.org/igwn.gwalert`, filled in by default since,
   unlike chatterbox's Rubin producer topic, this one is public and fixed).

`decam-chatterbox doctor` checks whether `~/.config/hop/auth.toml` exists.

**Real events are rare (tens to hundreds a year); mock events are not.** Both
GCN and SCIMMA send a sample alert for the mock superevent `MS181101ab` once
an hour, and GraceDB's own test infrastructure occasionally injects a `T...`
event. `Trigger.is_real` is `False` for both (checked from the
`superevent_id` prefix, which is the documented way to tell them apart: `S`
for a real superevent, `M` for mock/MDC, `T` for test). With
`ingest.allow_mock: true` (the default) they are still posted -- visibly
marked, and routed to `slack.test_channel` when one is configured -- which is
the easiest way to confirm the whole pipeline works without waiting for a
real candidate. Set it `false` to drop them instead.

The other ingest path, `files`, watches a directory for notice files (see
`scripts/fetch_samples.py` below) -- useful for testing without live SCIMMA
access, or for feeding in a notice saved by some other pipeline.

## Use

All commands below are shown as `python -m decam_chatterbox.cli`, which works
from the repository root whether or not the package is installed. Once
installed, the `decam-chatterbox` console script is equivalent.

Check the environment -- which capabilities work, which are degraded, and the
exact command to fix each one. Worth running first on any new host:

```bash
python -m decam_chatterbox.cli doctor
```

Confirm Slack credentials and channel access:

```bash
python -m decam_chatterbox.cli test-post
```

Download the official LVK sample notices (both JSON and Avro encodings, for
the mock superevent `MS181101ab`) for manual testing, since they are too
large to commit to the repository:

```bash
python scripts/fetch_samples.py
```

Check everything works without posting anything -- this is the main
development loop, and it renders the plot and prints the message as text:

```bash
python -m decam_chatterbox.cli replay tests/data/samples/MS181101ab-preliminary.json --dry-run
```

Run the service. It subscribes to SCIMMA by default; see
[Where alerts come from](#where-alerts-come-from):

```bash
python -m decam_chatterbox.cli serve
```

## Layout

```
decam_chatterbox/
  config.py       YAML + env settings
  models.py       Trigger, Localization, Geometry, ExternalCoincidence, GwEnrichment
  ingest/         notice decoding (JSON + Avro), transports (SCIMMA, files, replay),
                  the optional GraceDB chirp-mass lookup
  astro/          skymap credible regions, the CTIO almanac, accessible dark hours
  plots/          dark-hours map with the localization contour drawn on it
  slackbot/       Block Kit construction, delivery
  app.py          decode -> observability -> post, in one pass
  cli.py          serve | replay | test-post | doctor
scripts/
  fetch_samples.py  download the official LVK sample notices for manual testing
```

## Test

```bash
.venv/bin/python -m pytest
```

No network access required: notices are built programmatically at a small
`nside` from a real (if small) probability skymap written and read with
`ligo.skymap.io`, so the suite exercises the real decode path rather than a
hand-rolled substitute. Notable checks:

- The vectorized dark-hours altitude approximation is checked against
  astropy's own full `AltAz` transform at a handful of pixel/time pairs.
- A notice's `skymap` field arrives as raw bytes over SCIMMA (Avro) and
  base64-encoded text over GCN (JSON) -- both are decoded to the same
  localization.
- A retraction carries no `event` at all and must not be treated as a
  decode failure; a notice whose skymap fails to parse keeps every other
  field and says so, rather than losing the whole alert.
- A Burst event has no CBC classification or properties; a CBC event has no
  duration or central frequency. Neither should appear in the other's
  message.
- `superevent_id` prefixes `S`/`M`/`T` are asserted to map to `is_real`
  correctly, since that is what routes a notice to the test channel.

`tests/data/samples/MS181101ab-preliminary.json` (fetched with
`scripts/fetch_samples.py`, not committed) is worth a look once downloaded:
its localization -- a real, if antiquated, two-detector BAYESTAR
skymap -- sits almost entirely in the part of the sky that never clears
CTIO's airmass limit that night, so `replay --dry-run` on it prints a
`mean` accessible time and a `probability-weighted mean` that disagree by
four orders of magnitude. That gap is exactly the reason both numbers are
reported rather than one.

## Known limitations

**No template-coverage comparison, no planned follow-up, no simulation.**
Those are chatterbox's second stage, built around Rubin's own scheduler and
survey footprint; nothing analogous exists here, by design. If DECam
follow-up strategy or tiling ever needs automating, that is a second tool
built on top of this one's `Trigger`, not a feature added here.

**A Burst event's localization has no distance.** `distmean`/`diststd` are
only present in a CBC skymap's FITS header; the localization panel omits the
distance line rather than showing a stale or default value.

**The DECam-pointings figure is purely descriptive.** It divides the 90% area
by DECam's nominal 3.0 deg^2 field of view (Flaugher et al. 2015); it says
nothing about how those pointings would actually tile the region, since this
tool makes no tiling decision.

**GraceDB enrichment adds one number, not the whole event.** Unlike
chatterbox's Rubin-side enrichment -- which has to recover the entire
localization, distance, FAR and classification from GraceDB, because the
producer's own record carries none of them -- every one of those fields is
already on the notice here. `enrich.gracedb` fetches only the binned
chirp-mass estimate, and failure (no publication for this event, a GraceDB
outage, `ligo-gracedb` not installed) drops just that one line.
