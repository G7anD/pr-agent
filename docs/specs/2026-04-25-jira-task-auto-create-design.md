# PR Agent: auto-create Jira task when branch has no ticket ID

**Date:** 2026-04-25
**Author:** Erkin + assistant brainstorm
**Status:** Draft for user review

## Context

`pr-agent` already injects a Jira ticket link into the MR description whenever the branch name contains a recognizable Jira key:

- Regex: `\b[A-Z]{2,10}-\d{1,7}\b` in `pr_agent/tools/ticket_pr_compliance_check.py:find_jira_tickets`
- Hook: `pr_agent/tools/pr_description.py:135-145` — finds tickets in `branch_name`, prepends `> 🎯 **Jira Task:** [HIS-1234](https://jira.uicgroup.tech/browse/HIS-1234)` to the regenerated body.
- Config: `pr_description.jira_base_url = "https://jira.uicgroup.tech"`

What's **missing**: when the branch name has no key (e.g. `feature/login-bug-fix`), nothing is added. Developers either remember to embed `HIS-XXXX` in the branch name or the link never appears, and the work goes untracked in Jira.

We want the agent to **create the Jira task itself** in that case and inject the freshly-minted key into the MR description.

## Scope

**In scope:**
- A single new helper that, given branch name + MR metadata, returns a Jira ticket key — either an existing one already mentioned, or a freshly created one.
- Wire that helper into the existing `pr_description.py` injection point.
- Configuration to enable/disable the auto-create behavior and to hold the Jira REST credentials.
- Idempotency so re-running `/describe` on the same MR does not duplicate the task.

**Out of scope (deferred):**
- Multi-project routing (HD/HIS/etc. by repo). v1 is HIS-only.
- Component selection. v1 leaves component blank.
- Resolving GitLab username → Jira username via email lookup. v1 trusts the convention that `gitlab.username == jira.username`; falls back to unassigned on creation failure.
- Auto-labeling tasks as "auto-created-from-MR".
- Re-adoption of orphaned auto-created tasks.

## User-facing specification

### Trigger

Every time `/describe` runs (which happens automatically on MR open, and on demand via comment), the existing branch-key check fires first. If the branch contains a Jira key, behavior is unchanged.

### Auto-create flow (new)

When `find_jira_tickets(branch_name)` returns empty AND `pr_description.jira_auto_create_enabled = true`:

1. **Look at the live MR description** before the regenerated body replaces it. If `find_jira_tickets()` finds a key there (because a previous `/describe` run already created one), reuse that key — no creation, no duplication.
2. **Otherwise create a new Jira issue** via `POST /rest/api/2/issue`:
   - `project.key`: `HIS` (config-driven, but constant for now)
   - `issuetype.name`: `Task` (config-driven default)
   - `summary`: MR title verbatim
   - `assignee.name`: GitLab MR author username; if Jira rejects (`assignee invalid`), retry without assignee
   - `description`:
     ```
     Manba: <MR URL>

     <first 500 chars of MR description, trimmed>
     ```
   - No `components`, no `priority`, no `labels`. Triage stays manual.
3. **Inject the new key** into the regenerated PR body via the existing `> 🎯 Jira Task: [HIS-XXXX](...)` line.

### Failure modes

| Failure | Behavior |
|---|---|
| Auto-create disabled in config | Skip silently, fall through to old behavior (no link). |
| `pr_description.jira_auto_create_token` missing | Log WARNING once per process, skip. |
| Jira `POST /issue` returns 4xx (auth/permission) | Log ERROR with response body, skip injection, do NOT crash `/describe`. |
| Jira `POST /issue` returns 5xx or times out | Log ERROR, skip injection. Next `/describe` retries. |
| Assignee invalid → second attempt without assignee fails | Log ERROR, skip injection. |
| `git_provider.get_pr_description()` raises | Treat as empty description, proceed to creation (acceptable false-create risk). |

The pipeline never blocks the rest of `/describe` because of Jira issues. The MR body still gets the regular AI-generated description.

## Architecture

### New file: `pr_agent/tools/jira_auto_create.py`

```python
class JiraAutoCreateError(RuntimeError): ...

def resolve_existing_jira_ticket(branch_name: str, current_pr_body: str) -> Optional[str]:
    """First check branch, then current MR body. Return first ticket found, or None."""

def create_jira_task_for_mr(
    *,
    base_url: str,
    auth_token: str,
    project_key: str,
    issue_type: str,
    mr_title: str,
    mr_description: str,
    mr_url: str,
    mr_author: Optional[str],
    description_excerpt_chars: int = 500,
) -> str:
    """POST /rest/api/2/issue. Returns issue key (e.g. 'HIS-4801').
    Raises JiraAutoCreateError on failure. Retries once without assignee
    on 'assignee invalid'.
    """
```

The module uses `urllib` (project already uses it elsewhere). No new dependencies.

### Hook in `pr_description.py`

Replace lines 135-145 (the existing "Branch nomidan Jira ticket topib" block) with:

```python
jira_base_url = (get_settings().pr_description.get("jira_base_url", "") or "").rstrip("/")
if jira_base_url:
    branch_name = self.vars.get("branch", "") or ""
    current_body = ""
    try:
        current_body = self.git_provider.get_pr_description() or ""
    except Exception:
        pass
    ticket = resolve_existing_jira_ticket(branch_name, current_body)
    if not ticket and get_settings().pr_description.get("jira_auto_create_enabled", False):
        try:
            ticket = create_jira_task_for_mr(
                base_url=jira_base_url,
                auth_token=get_settings().pr_description.get("jira_auto_create_token", ""),
                project_key=get_settings().pr_description.get("jira_auto_create_project", "HIS"),
                issue_type=get_settings().pr_description.get("jira_auto_create_issue_type", "Task"),
                mr_title=pr_title,
                mr_description=current_body,
                mr_url=self.git_provider.get_git_url() or "",
                mr_author=self.vars.get("user", "") or self.vars.get("author", "") or "",
            )
        except JiraAutoCreateError as exc:
            get_logger().error(f"jira-auto-create failed: {exc}")
            ticket = None
    if ticket:
        link = f"[{ticket}]({jira_base_url}/browse/{ticket})"
        pr_body = "> 🎯 **Jira Task:** " + link + "\n\n" + pr_body
```

`pr_description.py` keeps its current responsibility (assemble the body); the `jira_auto_create.py` module owns the Jira HTTP details.

### Config additions to `pr_agent/settings/configuration.toml`

Inside `[pr_description]`:

```toml
jira_auto_create_enabled = true
jira_auto_create_project = "HIS"
jira_auto_create_issue_type = "Task"
# jira_auto_create_token must be set via env or .secrets.toml — never committed
```

Server-side, `jira_auto_create_token` is supplied via the existing pr-agent secret-loading path (env var override `PR_DESCRIPTION__JIRA_AUTO_CREATE_TOKEN` works with the dynaconf-style loader the project already uses for `GITLAB__PERSONAL_ACCESS_TOKEN`).

## Data flow

```
GitLab webhook (MR open)
   │
   ▼
gitlab_webhook.py → /describe pipeline
   │
   ▼
pr_description.PRDescription._run
   │ generates pr_title / pr_body via AI
   ▼
[NEW]  resolve_existing_jira_ticket(branch_name, current_mr_body)
   │      → branch hit? → return key
   │      → body hit?   → return key
   │      → none        → fall through
   ▼
[NEW]  create_jira_task_for_mr(...)
   │      POST /rest/api/2/issue → "HIS-4801"
   │      (single retry without assignee on failure)
   ▼
inject "> 🎯 Jira Task: [HIS-4801](...)" before pr_body
   │
   ▼
git_provider.publish_description(pr_title, final_body)
```

## Idempotency walk-through

1. **First /describe** on MR with branch `feature/login-bug`:
   - Branch: no key
   - Current MR body: empty (just opened) → no key
   - Create task → `HIS-4801`
   - Inject `> 🎯 Jira Task: [HIS-4801](...)` into body
2. **Second /describe** (e.g. user comments `/describe` 5 minutes later):
   - Branch: still no key
   - Current MR body: still has `[HIS-4801](...)` line from first run → `find_jira_tickets` returns `["HIS-4801"]`
   - Skip create; reuse `HIS-4801`; re-inject (no-op, line already there but the regenerator rewrites it cleanly)
3. **Manual override** (developer sets MR description to mention `HIS-9000` and `/describe` again):
   - Branch: no key
   - Current MR body: contains `HIS-9000` → use that
   - No new task created. `HIS-4801` is now orphaned in Jira and would need manual cleanup. Acceptable for v1.

## Testing

- **Unit, `tests/unittest/test_jira_auto_create.py`:**
  - `resolve_existing_jira_ticket("feature/HIS-1-x", "") == "HIS-1"`
  - `resolve_existing_jira_ticket("feature/x", "see HIS-7") == "HIS-7"`
  - `resolve_existing_jira_ticket("feature/x", "") is None`
  - `create_jira_task_for_mr` with mocked `urlopen`: builds correct payload (project, summary, assignee, description format), returns `result["key"]`.
  - Mocked Jira returns 400 "assignee not allowed" → retries without assignee, succeeds.
  - Mocked Jira returns 401/500 → raises `JiraAutoCreateError`.
- **Integration, run on the server:**
  - Create MR in `his/his` with branch `smoke/no-id-2026-04-25`, summary "Smoke test auto-create".
  - Trigger `/describe`.
  - Verify a new HIS-XXXX appears in Jira with summary matching MR title and description containing the MR URL.
  - Verify MR body has `> 🎯 Jira Task: [HIS-XXXX](...)` prepended.
  - Comment `/describe` again. Verify NO new Jira task created. Verify same `HIS-XXXX` still injected.
  - Delete the test MR and the test Jira task.

## Rollout

1. Land code + unit tests in one commit.
2. Set `jira_auto_create_enabled = false` initially in committed config — feature is dark.
3. On the server, set the token in `.secrets.toml` (or env) and flip `jira_auto_create_enabled = true`.
4. Smoke-test as above, then leave enabled.

## Open questions / future work

- **Component routing.** Currently triage assigns components manually. If the team consistently maps `his/his-support/*` paths to Support, v2 can derive component from the GitLab project path.
- **Reporter.** v1 lets Jira default to whoever the auth token belongs to (the bot account). v2 can set `reporter` to the MR author for accurate attribution if Jira instance allows that.
- **Stale auto-creates.** If a developer rewrites the MR description without preserving the injected line, `/describe` will create a second task. Future: keep a tiny `mr_id → jira_key` mapping in pr-agent state to detect that case.
- **GitLab username → Jira username mapping.** v1 assumes 1:1. We already maintain a separate mapping in `openclaw-jira-automation/config.users[]`. If those drift, swap the assignee-resolver to read the canonical mapping over HTTP from that service.
