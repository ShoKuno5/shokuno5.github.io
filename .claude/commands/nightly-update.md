# Nightly site update
You are the implementer agent. Perform these steps:

1. Pull latest main.
2. Create/update the site using `npm ci && npm run build`.
3. Run `npm run lint` and `npm run test`.
4. Commit changes with message "Nightly update $DATE".
5. Push to current branch.
