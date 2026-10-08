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
- **A high-priority triage badge**, when `PriorityConfig`'s thresholds on FAR,
  90% area and NS-classification probability are all met, with the measured
  value and threshold for each one spelled out underneath rather than left as
  a bare badge. This never changes what gets posted -- every real and mock
  alert still reaches `slack.channel` -- it only adds the badge, gates
  `slack.mention`, and controls an additional cross-post to
  `slack.urgent_channel`. See [Priority triage](#priority-triage).

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

## Priority triage

`priority` in `config.yaml` sets three thresholds -- an alert is high priority
when every applicable one is met:

- `max_far_hz`: false-alarm rate. Defaults to `3.17e-8` ("about 1 per year",
  the same convention chatterbox's own Rubin gold/silver GW classes use).
- `max_area_deg2`: 90% credible area. Defaults to `500`.
- `min_ns_classification`: minimum p(BNS) + p(NSBH) -- a merger likely to
  involve a neutron star, and so plausibly EM-bright. Defaults to `0.5`.
  Exempted for a Burst event, which carries no classification at all; a
  retraction and anything that is not a real superevent are never high
  priority regardless of the numbers.

This is triage, not a decision: it never changes what gets posted to
`slack.channel` -- every real and mock alert goes there either way, same as
before. What it does control:

- A `:rotating_light: HIGH PRIORITY` badge on the message, with the measured
  value and threshold for each criterion spelled out underneath (e.g. `FAR
  9.11e-14 Hz <= 3.17e-08 Hz`), so it is never a bare, unexplained flag.
- `slack.mention` is included only on a high-priority alert's cross-post (see
  next point) -- not on every post, so a routine BBH detection does not page
  anyone.
- If `slack.urgent_channel` is set, a high-priority alert is *also* posted
  there (with the mention), in addition to `slack.channel`. This is a
  cross-post, not a redirect: losing it (a rare Slack API failure) loses only
  the extra visibility, never the alert itself.

**Testing the page itself.** `test-ping` posts a clearly-marked "TEST ping --
no action needed" message that @-mentions `slack.mention`, to
`slack.urgent_channel` (or `slack.channel` if that is empty), so the on-call
path can be checked without waiting for a real high-priority alert. It really
notifies whoever is listed, so it prints who and where and asks first (skip
with `--yes`). To rehearse without paging anyone, point it at yourself:

```bash
python -m decam_chatterbox.cli test-ping --mention U0123ABCD --channel "#my-test-channel"
```

Find a user ID under *Profile -> three dots -> Copy member ID*, and a user
group's under *People -> User groups* (`!subteam^S...`). The mention is also
rendered as a visible first block, not just carried in the notification text.

Set `priority.enabled: false` to turn the whole feature off -- no badge, no
mention, no cross-post, ever, regardless of the thresholds.

## Heartbeat and shutdown notice

Both apply only to `decam-chatterbox serve` -- a `replay` is a one-shot
inspection, not a long-running service, so neither is relevant to it.

**`heartbeat.enabled` (default true) posts a short "still running" message
every `heartbeat.interval_s` (default 3600, hourly)**, on its own background
thread so it keeps going even between alerts: uptime, the ingest source it is
watching, and how many alerts it has handled and when it last handled one.
The failure posts elsewhere in this tool only fire because something raised;
a hung process, or a SCIMMA connection that has quietly died without an
exception, would otherwise be indistinguishable from a quiet night. The
heartbeat is what rules that out. It posts to `slack.heartbeat_channel`
(falling back to `slack.channel`, including when that channel cannot be
reached).

**`serve` also posts once when it starts, and once when it stops, in the same
style.** The start post is also the immediate check on `slack.heartbeat_channel`:
a misspelled channel, or a private one the bot was never invited to (Slack
reports both as `channel_not_found`), would otherwise only show up when the
first heartbeat came due an hour later. If the heartbeat channel cannot be
posted to, the heartbeat, start and stop posts all fall back to `slack.channel`
with the Slack error attached, rather than becoming a log line nobody reads.
Fix it by inviting the bot (`/invite @<bot name>`) or by setting the channel's
ID (`C0123456789`) instead of its name.

The stop post is sent on Ctrl-C, on `SIGTERM` (`kill`, `systemctl stop`,
`docker stop`, a host shutdown) and on `SIGHUP` (a closed ssh or tmux session,
which kills a process instantly by default; a `nohup`'d process keeps ignoring
it, as the operator asked). All three are installed to behave exactly like
Ctrl-C for the duration of the run, and once the stop post has started, further
`SIGTERM`/`SIGHUP` are ignored so a host that sends both in quick succession
does not interrupt the post part-way. It is also posted when the ingest source
itself fails (which already posts as a failure -- see
[What it posts](#what-it-posts)). Both read the same status line as the
heartbeat, so the channel can tell "stopped on purpose" from "crashed" or
"hung" without cross-referencing a log.

Nothing can catch `SIGKILL` (`kill -9`), which is also what a service manager
sends when a stop takes longer than its timeout, or what the kernel's
out-of-memory killer sends. A missing stop post together with a stopped
heartbeat means one of those, or a power loss. Running under systemd gives the
most reliable stop, since it sends `SIGTERM` first and waits:

```ini
[Service]
ExecStart=/path/to/.venv/bin/python -m decam_chatterbox.cli serve
WorkingDirectory=/path/to/decam-chatterbox
Restart=on-failure
KillSignal=SIGTERM
TimeoutStopSec=60
```

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

**`KafkaError{code=GROUP_AUTHORIZATION_FAILED, ...}` at startup** means
SCIMMA's Kafka ACLs rejected the consumer group id, not the credential or the
topic subscription -- those fail differently (see below). SCIMMA only
authorizes a consumer group that matches your own credential's username, so
`ingest.hop_group_id` defaults to empty and is derived automatically as
`{your hop username}-decam-chatterbox`, which satisfies that rule and (unlike
leaving it to hop-client's own fallback, a fresh random suffix every time) is
stable across restarts, so Kafka still remembers the offset. This should not
come up with the default; if it does, `hop auth list` will show whether a
credential for `kafka.scimma.org` is present at all, and the log line just
above the traceback names the group id decam-chatterbox actually tried to
use.

**`TOPIC_AUTHORIZATION_FAILED`** instead means step 3 was missed or has not
propagated yet -- the credential exists but isn't subscribed to
`igwn.gwalert`, or the up-to-an-hour delay after subscribing hasn't passed.

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
  priority.py     high-priority triage: FAR, area and NS-classification thresholds
  heartbeat.py    the hourly "still running" post and the shutdown notice (serve only)
  slackbot/       Block Kit construction, delivery
  app.py          decode -> observability -> priority -> post, in one pass
  cli.py          serve | replay | test-post | test-ping | doctor
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
