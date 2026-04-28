# GitHub + production Fly org (one-time)

## 1) Push this repo to `github.com/goenablers/page-orientation-api`

The machine used for implementation may not have the `gh` CLI. Use either:

### Option A: GitHub CLI

```bash
brew install gh
gh auth login
cd /path/to/this/repo
git remote add origin https://github.com/goenablers/page-orientation-api.git   # or SSH URL
git push -u origin master
git checkout -b release
git push -u origin release
```

Create the empty repository in the `goenablers` org first (GitHub UI) if it does not exist, then add `origin` and push.

### Option B: GitHub UI

Create **New repository** under `goenablers` → `page-orientation-api` (no README). Then:

```bash
git remote add origin https://github.com/goenablers/page-orientation-api.git
git push -u origin master
git push -u origin release   # after: git checkout -b release
```

## 2) GitHub Actions secrets

In the repo: **Settings → Secrets and variables → Actions**, add:

| Name | Value |
|------|--------|
| `FLY_API_TOKEN_STG` | Output of `fly tokens create deploy -a page-orientation-api-stg` |
| `FLY_API_TOKEN_PRD` | Output of `fly tokens create deploy -a page-orientation-api-prd` (after the PROD app exists) |

Pushes to `release` / `master` will deploy per [.github/workflows/deploy.yml](../.github/workflows/deploy.yml).

## 3) Production Fly org `prd-goenablers`

If `fly orgs list` does not show `prd-goenablers`, create the org in the [Fly.io dashboard](https://fly.io/dashboard) or get invited, then:

```bash
fly apps create page-orientation-api-prd --org prd-goenablers
fly deploy -c fly.production.toml --remote-only
```

## 4) Staging URL (already deployed in this project)

- **STG:** `https://page-orientation-api-stg.fly.dev`

When PROD is created, it will be: `https://page-orientation-api-prd.fly.dev`
