import os
import time
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import logging
from github import Auth, Github, RateLimitExceededException, GithubException


load_dotenv()

ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
logging.info(f"Project root: {PROJECT_ROOT}")
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "github_repos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

auth = Auth.Token(ACCESS_TOKEN)
gh = Github(auth=auth)

MAX_REPOS = 10
cloned_count = 0

def safe_clone(repo_url: str, target_dir: str) -> bool:
    """Shallow clone a repo if not already exists"""
    if os.path.exists(target_dir):
        logging.info(f"[SKIP] {target_dir} already exists")
        return False
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],  # get latest commit only
            check=True,
        )
        logging.info(f"[CLONED] {repo_url} to {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"[ERROR] cloning {repo_url}: {e}")
        return False
        
def clone_repos():
    global cloned_count, MAX_REPOS
    query = f"language:Python stars:>=1000 license:mit"
    repos = gh.search_repositories(query=query, sort="stars", order="desc")
    
    logging.info(f"Found ~{repos.totalCount} repos")
    
    for repo in repos:
        if cloned_count >= MAX_REPOS:
            logging.info(f"Reached max repos limit: {MAX_REPOS}")
            break
        
        try:
            # Get the core API rate limit info (for REST API calls)
            rate_limit_overview = gh.get_rate_limit()
            core_rate = rate_limit_overview.resources.core
            
            # If there are fewer than 10 requests left in the current window
            if core_rate.remaining < 10:
                # Calculate how many seconds until the limit resets
                reset_time = core_rate.reset.timestamp() - time.time()
                
                # Add a 5 second buffer to ensure the reset has happened
                wait_for = max(0, int(reset_time) + 5)
                
                # Log and sleep until the rate limit resets
                logging.info(f"Rate limit hit. Sleep {wait_for}s")
                time.sleep(wait_for)

            target_dir = os.path.join(OUTPUT_DIR, repo.full_name.replace("/", "_"))
            cloned_success = safe_clone(repo.clone_url, target_dir)

            if cloned_success:
                cloned_count += 1
                logging.info(f"Cloned {cloned_count}/{MAX_REPOS}: {repo.full_name}")
                
            time.sleep(2)  # gentle delay to avoid overload

        except RateLimitExceededException:
            print("Rate limit exceeded. Sleeping 1 min")
            time.sleep(60)
        except GithubException as e:
            print(f"[GH ERROR] {repo.full_name}: {e}")
        except Exception as e:
            print(f"[ERROR] {repo.full_name}: {e}")

if __name__ == "__main__":
    clone_repos()
    