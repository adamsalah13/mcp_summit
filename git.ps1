# 1) Make the first commit
git commit -m "Clean start – working Summit MCP stack"

# 2) Rename the branch to 'main'
git branch -M main

# 3) Add (or set) the remote
git remote add origin https://github.com/adamsalah13/mcp_summit.git
# if it already exists but points elsewhere:
# git remote set-url origin https://github.com/adamsalah13/mcp_summit.git

# 4) Push, overwriting the remote branch
git push --force origin main        # --force-with-lease is fine too
