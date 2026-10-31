"""Where the app keeps its files, and the addresses it answers to.

**No BRAIN credentials here.** They are typed into the sign-in screen and sealed in the
local vault, and nothing reads them from the environment: seeding them from a file would
let a checked-out repository sign in as its owner.
"""

from dataclasses import dataclass
from pathlib import Path

#: The pages that may open the telemetry socket: the Vite dev server, and the built UI the
#: backend serves itself.
ALLOWED_ORIGINS = frozenset(
    {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    }
)


#: Home-relative. Must stay on native ext4 — see CLAUDE.md.
DATA_DIR = Path.home().joinpath(".alpha-harness").resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Where local state lives."""

    data_dir: Path = DATA_DIR
    #: A consultant's daily simulation allowance, shown until the day's first simulation POST
    #: returns its x-ratelimit-* headers.
    daily_simulation_allowance: int = 5000
    #: What the quota gate lets the app spend in a platform day. Never above the allowance.
    #: Raised once the gate counted by BRAIN's own quota headers: a local count ran ~800 high.
    simulation_operating_ceiling: int = 5000
    #: Part of the operating ceiling that ordinary requests cannot touch: kept for your own
    #: simulations on BRAIN.
    simulation_protected_reserve: int = 300
    #: An unsettled reservation older than this is marked uncertain on restart, never released.
    reservation_stale_seconds: float = 900.0

    @property
    def sqlite_path(self) -> Path:
        """Operational state: credentials, sessions, simulation records, templates."""
        return self.data_dir / "harness.db"

    @property
    def duckdb_path(self) -> Path:
        """Analytical store: the data-field catalog."""
        return self.data_dir / "catalog.duckdb"

    @property
    def key_path(self) -> Path:
        """AES-GCM master key. Created 0600 on first run."""
        return self.data_dir / "key"

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
