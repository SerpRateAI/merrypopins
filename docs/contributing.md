# Contributing

Contributions are welcome! Please file issues and submit pull requests on [GitHub](https://github.com/SerpRateAI/merrypoppins).

## Branching Model

main ← 📦 production releases
dev ← 🛠 active development (default Pull Request target)

* **`main`** holds only stable, version-tagged releases.  
* **`dev`** is the rolling integration branch where all feature / fix PRs land first.  
  Maintainers periodically open an *internal* PR from **`dev` → `main`** when a new
  release is ready.

## How to Open a Pull Request

1. **Fork** the repository to your GitHub account.  
2. **Clone** your fork and set the upstream remote:  
   ```bash
   git clone https://github.com/<your-user>/merrypopins.git
   cd merrypopins
   git remote add upstream https://github.com/SerpRateAI/merrypopins.git
   ```
3. **Sync & branch off** `dev`:
   ```bash
   git fetch upstream
   git checkout -b feature/awesome upstream/dev
   ```
4. Do your work → **commit**:
   ```bash
   git commit -m "feat: add awesome feature"
   ```
5. **Push** to your fork:
   ```bash
   git push origin feature/awesome
   ```
6. Open a **pull request _into_ `dev`** (set the PR’s base branch to `dev`).  
7. Address any review comments & keep your feature branch updated with the latest `dev` if needed.  

> **Note:** Once your PR is merged into `dev`, the maintainers will handle promoting `dev` to `main` when preparing a new release—please don’t open PRs directly against `main`.

---

This project is licensed under the MIT see [LICENSE](LICENSE).