import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client | None = None


def get_supabase() -> Client:
    """Return a singleton Supabase client with startup validation."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")

        # ── Validate env vars before attempting to connect ──────────────
        if not url:
            raise RuntimeError(
                "SUPABASE_URL is not set. "
                "Add it to Railway Variables tab."
            )
        if not key:
            raise RuntimeError(
                "SUPABASE_SERVICE_KEY is not set. "
                "Add the service_role key to Railway Variables tab."
            )
        if not url.startswith("https://"):
            raise RuntimeError(
                f"SUPABASE_URL must start with 'https://'. Got: '{url[:40]}'. "
                "Copy the Project URL from Supabase → Settings → API."
            )
        if not url.endswith(".supabase.co"):
            raise RuntimeError(
                f"SUPABASE_URL must end with '.supabase.co'. Got: '{url[:40]}'. "
                "Copy the Project URL from Supabase → Settings → API. "
                "It looks like https://xxxxxxxxxxx.supabase.co"
            )

        _client = create_client(url, key)
    return _client
