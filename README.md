# Transaction Fraud Detection Project

## Project Overview
This project aims to detect fraudulent transactions using an imbalanced dataset. The dataset consists of transaction records labeled as fraud (`1`) or non-fraud (`0`).

## Setup

### Install Git LFS
To handle large files in this project, you need to set up Git LFS. Follow these steps:

1. **Install Git LFS**:
   - For macOS (using Homebrew):
     ```bash
     brew install git-lfs
     ```
   - For Windows, you can download the installer from the [Git LFS website](https://git-lfs.github.com/).
   - For Linux, you can follow the instructions on the [Git LFS installation page](https://github.com/git-lfs/git-lfs/wiki/Installation).

2. **Initialize Git LFS in your repository**:
   After installing Git LFS, navigate to your project directory and run:
   ```bash
   git lfs install
   ```

3. Track large files: 
    To track specific file types (e.g., CSV files), use the following command:
    ```bash
    git lfs track "*.csv"
    ```

## Install Dependencies
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```
