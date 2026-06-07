Clone this repository and upload your tasks on a new branch with your name.

If you already have a repository, create a branch on your repository and set it as the remote for this repository. Then push your branch to the remote.

### 1. You've cloned a repository you've made on github:
```bash
git remote set-url "origin" https://github.com/Epoch-2026-2027/Learning_Phase.git  # Sets your repository remote as the Learning Phase repository
git checkout -b <your-name>  # Creates and switches to a new branch with your name
git push -u origin <your-name>  # Pushes your branch to the remote repository
```

### 2. You've initialized a local repository:
```bash
git remote add origin https://github.com/Epoch-2026-2027/Learning_Phase.git  # Sets your repository remote as the Learning Phase repository
git checkout -b <your-name>  # Creates and switches to a new branch with your name
git push -u origin <your-name>  # Pushes your branch to the remote repository
```
Note you'll most likely have to resolve merge conflicts if you have an existing repository. Make sure to pull the latest changes from the Learning Phase repository before pushing your branch to avoid conflicts. (use rebase to resolve conflicts if you have an existing repository)

If you don't have an existing repository, clone this repository and create a new branch with your name:
```bash
git clone https://github.com/Epoch-2026-2027/Learning_Phase.git
cd Learning_Phase
git checkout -b <your-name>  # Creates and switches to a new branch with your name
# Now upload/move your task files
git push -u origin <your-name>  # Pushes your branch to the remote
```
