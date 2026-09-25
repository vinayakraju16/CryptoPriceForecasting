# Security

## Reporting

Open an issue or contact the maintainer directly. Please do not open a public
issue for a live credential leak.

---

## Known exposure: rotate these credentials

The following values were **committed in plaintext** to this public repository
in `reddit_sentiment_analysis.ipynb`:

- Reddit `client_id` beginning `h1oT9EC...`
- Reddit `client_secret` beginning `wq3QJI...`

Removing them from the working tree does **not** remove them from git history,
and forks and clones may still hold them. **They must be revoked and regenerated
in the Reddit app console** (`https://www.reddit.com/prefs/apps`). Until that is
done, treat them as public.

The notebook has not been rewritten in place because the file is historical
evidence; the replacement, `scripts/fetch_reddit_sentiment.py`, reads
credentials from the environment and fails loudly when they are absent.

## Handling credentials in this project

- Never commit credentials, tokens or API keys. `.gitignore` excludes `.env`.
- Copy `.env.example` to `.env` and fill it locally.
- The reddit script exits immediately if any of `REDDIT_CLIENT_ID`,
  `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` is missing - there is no
  fallback to a default value, by design.
- If a credential is ever committed, rotate it first, then clean history.

## Web application posture

- Binds `127.0.0.1` by default; set `HOST` deliberately if you need otherwise.
- Debug mode is opt-in (`FLASK_DEBUG=1`) and never on by default - the Werkzeug
  debugger executes arbitrary code from the browser.
- Request bodies are capped by `MAX_CONTENT_LENGTH` (8 MB default).
- Error responses never include internal paths or raw exception text.
- The app serves only templates and static assets from its own directories; it
  does not read files chosen by the client, and `_symbol_from_model_name`
  strips any path components from user input.
